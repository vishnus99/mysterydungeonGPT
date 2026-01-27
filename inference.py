import subprocess
import time
import os
import modal
import sys
import re
import json
from openai import OpenAI

#========================================SERVER-SIDE====================================================
sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime")
    .entrypoint([])
    .pip_install(
        "huggingface-hub", 
        "fastapi", 
        "httpx", 
        "pydantic>=2.0",
        "torch",
        "transformers",
        "peft",
        "accelerate"
    )
    .add_local_file("qwen3_nonthinking.jinja", "/qwen3_nonthinking.jinja")
)

app = modal.App("mystery-dungeon-inference-chat")
volume = modal.Volume.from_name("mystery-dungeon-models", create_if_missing=True)

GPU_TYPE, N_GPUS = "A100", 1
GPU = f"{GPU_TYPE}:{N_GPUS}"
PORT = 8000
MODEL_PATH = "/models/merged_model"

@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    container_idle_timeout=300
)

class SGLangServer:
    @modal.enter()
    def startup(self):
        import sys
        
        # Merge model if not cached
        if not os.path.exists(f"{MODEL_PATH}/config.json"):
            print("=== MERGING LORA INTO BASE MODEL ===", flush=True)
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            peft_model = PeftModel.from_pretrained(base_model, "vishnusm/mysterydungeonGPT")
            merged_model = peft_model.merge_and_unload()
            
            os.makedirs(MODEL_PATH, exist_ok=True)
            merged_model.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(MODEL_PATH)
            volume.commit()
            print("=== MODEL MERGED AND SAVED ===", flush=True)
        else:
            print("=== USING CACHED MERGED MODEL ===", flush=True)
        
        # Launch SGLang with merged model (no LoRA)
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", MODEL_PATH,
            "--host", "0.0.0.0",
            "--port", str(PORT),
            "--disable-cuda-graph",
            "--chat-template", "/qwen3_nonthinking.jinja"
        ]
        print(f"Command: {' '.join(cmd)}", flush=True)
        
        self.process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        self._wait_ready()
    
    def _wait_ready(self, timeout=180):
        import requests
        print(f"=== WAITING FOR SERVER ===", flush=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = requests.get(f"http://localhost:{PORT}/health", timeout=2)
                if response.status_code == 200:
                    print("=== SGLANG SERVER READY ===", flush=True)
                    return
            except:
                pass
            time.sleep(2)
        raise RuntimeError("SGLang server failed to start")
    
    @modal.exit()
    def stop(self):
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process.wait()

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, Request, Response
        import httpx
        
        web_app = FastAPI()
        
        @web_app.get("/")
        async def root():
            return {"status": "ok"}
        
        @web_app.get("/get-url")
        async def get_url(request: Request):
            url = str(request.base_url).rstrip('/')
            return {"url": url}
        
        @web_app.api_route("/{path:path}", methods=["GET", "POST"])
        async def proxy(request: Request, path: str):
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.request(
                    request.method,
                    f"http://localhost:{PORT}/{path}",
                    content=await request.body(),
                    headers={k: v for k, v in request.headers.items() if k.lower() != "host"}
                )
                return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
        
        return web_app


#==============================================CLIENT SIDE===================================================

class InferenceClient:
    def __init__(self, url=None):
        if url is None:
            raise ValueError("url parameter is required")
        self.url = url.rstrip('/')
        self.client = OpenAI(
            base_url=f"{self.url}/v1",
            api_key="not-needed"
        )

    @classmethod
    def from_modal_app(cls, app_name="mystery-dungeon-inference-chat"):
        try:
            function = modal.Function.from_name(
                app_name,
                "SGLangServer.serve"
            )
            url = function.get_web_url().rstrip('/')
            return cls(url)
        except Exception as e:
            raise RuntimeError(f"Failed to get URL from Modal: {e}")
    
    @staticmethod
    def extract_json_from_text(text):
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return None

        json_str = json_match.group()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix trailing commas
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            try:
                return json.loads(json_str)
            except:
                return None

    def generate_map(self, prompt, **kwargs):
        response = self.client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.8),
            top_p=kwargs.get("top_p", 0.9),
            max_tokens=kwargs.get("max_new_tokens", 6000),
            stream=False
        )

        generated_text = response.choices[0].message.content

        # Extract JSON client-side
        map_json = self.extract_json_from_text(generated_text)

        return {
            "raw_text": generated_text,
            "map_json": map_json,
            "success": map_json is not None
        }





