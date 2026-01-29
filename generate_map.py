import sys
import re
import json
from openai import OpenAI
from pathlib import Path
from datetime import datetime

class MapGenerator:
    MODAL_URL = "" #provided by Modal upon server setup

    def __init__(self):
        self.client = OpenAI(
            base_url=f"{self.MODAL_URL}/v1",
            api_key="not_needed"
        )
    
    @staticmethod
    def extract_json_from_text(text):
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            return None
        
        json_str = json_match.group()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*}', ']', json_str)
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
            max_tokens=kwargs.get("max_new_tokens", 10000),
            stream=False
        )

        generated_text = response.choices[0].message.content

        map_json = self.extract_json_from_text(generated_text)

        return {
            "raw_text": generated_text,
            "map_json": map_json,
            "success": map_json is not None
        }

    def _get_maps_directory(self):
        script_dir = Path(__file__).parent
        maps_dir = script_dir / "web_game" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        return maps_dir
    
    def _generate_map_id(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"map_generated_{timestamp}"
    
    def save_map(self, map_json, map_id=None):
        if map_id is None:
            map_id = self._generate_map_id()
        
        maps_dir = self._get_maps_directory()
        file_path = maps_dir / f"{map_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(map_json, f, indent=2)
        
        return map_id, file_path

    def update_map_index(self, map_id):
        maps_dir = self._get_maps_directory()
        index_path = maps_dir / "map_index.json"
        
        # Load existing index or create new one
        if index_path.exists():
            with open(index_path, 'r') as f:
                map_index = json.load(f)
        else:
            map_index = {
                "total_maps": 0,
                "map_ids": [],
                "source": "generated",
                "split": "generated"
            }
        
        # Add map_id if not already present
        if map_id not in map_index['map_ids']:
            map_index['map_ids'].append(map_id)
            map_index['total_maps'] = len(map_index['map_ids'])
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(map_index, f, indent=2)
        
        return map_index

    def save_and_index_map(self, map_json, map_id=None):
        map_id, file_path = self.save_map(map_json, map_id)
        map_index = self.update_map_index(map_id)
        return map_id, file_path, map_index

    

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Generate a medium difficulty dungeon with 8 rooms"
    generator = MapGenerator()
    result = generator.generate_map(prompt)
    print(result)

    if result["success"]:
        map_id, file_path, map_index = generator.save_and_index_map(result["map_json"])
    else:
        print("Failed to generate map")
