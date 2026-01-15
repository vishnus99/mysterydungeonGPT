import asyncio
import json
import subprocess
import time
import aiohttp
import modal
import modal.experimental


MINUTES = 60

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime").entrypoint([])
).pip_install(
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "pybase64",
    "sglang",
    "requests",
    "numpy",
    "huggingface-hub",
    "requests"
).apt_install("git")

sglang_image = sglang_image.add_local_python_source("mysterydungeonGPT")

app = modal.App("mystery-dungeon-inference-chat")

volume = modal.Volume.from_name("mystery-dungeon-models", create_if_missing=True)
volume_path = "/models/merged_model"

GPU_TYPE, N_GPUS="A100", 1
GPU = f"{GPU_TYPE}:{N_GPUS}"
PORT = 8000

@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={"/model": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    container_idle_timeout=300
)
@modal.experimental.http_server(
    port=PORT,
    exit_grace_period=5
)

class SGLangServer:
    @modal.enter()
    def startup(self):
        import os

        cache_path = "/models/merged_model"
        if not os.path.exists(cache_path) or not os.path.exists(f"{cache_path}/config.json"):
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

            os.makedirs(cache_path, exist_ok=True)
            merged_model.save_pretrained(cache_path)
            tokenizer.save_pretrained(cache_path)
            volume.commit()
        
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", cache_path,
            "--host", "0.0.0.0",
            "--port", str(PORT)
        ]

        self.process = subprocess.Popen(cmd)
        self._wait_ready()
    
    def _wait_ready(self, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = requests.get(f"http://localhost:{PORT}/health", timeout=2)
                if response.status_code == 200:
                    print("SGLang server ready")
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

@app.function(
    image=sglang_image,
    gpu=GPU,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface")]
)
@modal.web_endpoint(method="POST")
def generate(
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_new_tokens: int = 6000,
    repetition_penalty: float = 1.1,
    do_sample: bool = True
):
    import requests
    from transformers import AutoTokenizer

    sglang_url = SGLangServer._experimental_get_flash_urls()[0]

    if not hasattr(generate, "_tokenizer"):
        generate._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer = generate._tokenizer

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    payload = {
        "model": "merged_model",
        "messages": [{"role": "user", "content": formatted_prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "stream": False
    }

    response = requests.post(
        f"{sglang_url}/v1/chat/completions",
        json=payload,
        timeout=120
    )
    response.raise_for_status()

    result = response.json()
    generated_text = result["choices"][0]["message"]["content"]

    return {"text": generated_text}