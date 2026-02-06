import sys
import re
import json
from openai import OpenAI
from pathlib import Path
from datetime import datetime

from mysterydungeonGPT.helpers import coordinates_to_grid

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

    def convert_to_game_format(self, map_json):
        """Convert coordinate-based model output to game format (tiles grid, spawns, enemies)."""
        if not map_json or "walkable_tiles" not in map_json:
            return None
        width = int(map_json.get("width", 56))
        height = int(map_json.get("height", 32))
        walkable = map_json.get("walkable_tiles", [])
        valid_coords = []
        for c in walkable:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                x, y = int(c[0]), int(c[1])
                if 0 <= x < width and 0 <= y < height:
                    valid_coords.append([x, y])
        full_grid = coordinates_to_grid(valid_coords, width=width, height=height)
        game_tiles = full_grid.copy()

        player_spawn = map_json.get("player_spawn", [0, 0])
        stairs_spawn = map_json.get("stairs_spawn", [0, 0])
        if not isinstance(player_spawn, (list, tuple)) or len(player_spawn) < 2:
            player_spawn = [0, 0]
        if not isinstance(stairs_spawn, (list, tuple)) or len(stairs_spawn) < 2:
            stairs_spawn = [0, 0]
        px, py = int(player_spawn[0]), int(player_spawn[1])
        sx, sy = int(stairs_spawn[0]), int(stairs_spawn[1])

        def nearest_floor(tx, ty):
            nearest = None
            min_dist = float("inf")
            for y in range(height):
                for x in range(width):
                    if game_tiles[y, x] == 1:
                        d = abs(x - tx) + abs(y - ty)
                        if d < min_dist:
                            min_dist, nearest = d, [x, y]
            return nearest

        if not (0 <= px < width and 0 <= py < height) or game_tiles[py, px] == 0:
            n = nearest_floor(px, py)
            if n:
                px, py = n[0], n[1]
        if not (0 <= sx < width and 0 <= sy < height) or game_tiles[sy, sx] == 0:
            n = nearest_floor(sx, sy)
            if n:
                sx, sy = n[0], n[1]

        game_tiles[py, px] = 1
        game_tiles[sy, sx] = 1

        game_enemies = []
        for enemy in map_json.get("enemies", []):
            if isinstance(enemy, dict) and "x" in enemy and "y" in enemy:
                game_enemies.append({"x": int(enemy["x"]), "y": int(enemy["y"])})
            elif isinstance(enemy, (list, tuple)) and len(enemy) >= 2:
                game_enemies.append({"x": int(enemy[0]), "y": int(enemy[1])})

        return {
            "tiles": game_tiles.tolist(),
            "player_spawn": [px, py],
            "stairs_spawn": [sx, sy],
            "width": width,
            "height": height,
            "difficulty": map_json.get("difficulty", "medium"),
            "enemies": game_enemies,
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
        game_map = generator.convert_to_game_format(result["map_json"])
        if game_map:
            map_id, file_path, map_index = generator.save_and_index_map(game_map)
            print(f"Map saved: {file_path}")
        else:
            print("Failed to convert map to game format")
    else:
        print("Failed to generate map")
