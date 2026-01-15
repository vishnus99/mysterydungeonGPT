from openai import OpenAI
from transformers import AutoTokenizer
from mysterydungeonGPT.helpers import extract_json_from_text

# SGLang server URL (get from Modal deployment)
SGLANG_URL = "https://vishnus99--mystery-dungeon-inference-chat-generate.modal.run"

# Initialize client
client = OpenAI(
    base_url=f"{SGLANG_URL}/v1",
    api_key="not-needed"
)

# Load tokenizer for chat template
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def generate_map(prompt, **kwargs):
    """Generate map using OpenAI-compatible API"""
    
    # Format prompt with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Call SGLang via OpenAI client
    response = client.chat.completions.create(
        model="merged_model",
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=kwargs.get("temperature", 0.8),
        top_p=kwargs.get("top_p", 0.9),
        max_tokens=kwargs.get("max_new_tokens", 6000),
        stream=False
    )
    
    generated_text = response.choices[0].message.content
    
    # Extract JSON client-side
    map_json = extract_json_from_text(generated_text)
    
    return {
        "raw_text": generated_text,
        "map_json": map_json,
        "success": map_json is not None
    }

# Usage
result = generate_map("Generate a medium difficulty dungeon with 8 rooms")
print(result)