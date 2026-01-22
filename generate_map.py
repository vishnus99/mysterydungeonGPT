import sys
import re
import json
from openai import OpenAI

# Modal URL (get from deployment)
MODAL_URL = "https://jesterlabs--mystery-dungeon-inference-chat-sglangserver-serve.modal.run"

# Initialize client
client = OpenAI(
    base_url=f"{MODAL_URL}/v1",
    api_key="not-needed"
)

def extract_json_from_text(text):
    """Extract JSON object from text."""
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

def generate_map(prompt, **kwargs):
    """Generate map using OpenAI-compatible API"""
    
    response = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": prompt}],
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

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Generate a medium difficulty dungeon with 8 rooms"
    result = generate_map(prompt)
    print(result)
