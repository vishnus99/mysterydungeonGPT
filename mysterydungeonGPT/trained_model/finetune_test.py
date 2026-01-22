#!/usr/bin/env python3
"""
Fine-tuned Model Test Script
Converted from finetune_test.ipynb

Usage:
    python finetune_test.py [prompt]
    
Examples:
    python finetune_test.py "Generate a medium difficulty dungeon with 6 rooms"
    python finetune_test.py  # Uses default prompt
"""

import sys
import argparse

def main(prompt=None):
    """Main test function"""
    # Cell 0
    # Test Fine-Tuned Model with Coordinate-Based Format
    import sys
    import os
    import json
    import re
    import numpy as np
    from pathlib import Path

    # Add project root to path
    # Get script directory and calculate project root (2 levels up)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch

    print("Imports successful")
    print(f"Project root: {project_root}")


    # Cell 1
    # Load the fine-tuned model
    # Use absolute path relative to script location (script_dir already defined above)
    finetuned_model_path = script_dir / "final_model"

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    # Device selection: Avoid MPS on macOS due to memory limitations
    if torch.cuda.is_available():
        device_map = "auto"
        print("Using CUDA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS has memory limitations, use CPU instead for stability
        device_map = "cpu"
        print("MPS available but using CPU for stability (MPS has memory limitations)")
    else:
        device_map = "cpu"
        print("Using CPU")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=torch.float16,
        device_map=device_map
    )

    ft_model = PeftModel.from_pretrained(base_model, "vishnusm/mysterydungeonGPT")
    ft_model.eval()

    print("Model loaded successfully")


    # Cell 2
    # Import helper functions
    from mysterydungeonGPT.helpers import coordinates_to_grid, extract_json_from_text

    print("Helper functions imported")


    # Cell 3
    # Create test prompt (use provided prompt or default)
    if prompt is None:
        test_prompt = "Generate a medium difficulty dungeon with 8 rooms"
    else:
        test_prompt = prompt

    print(f"Test prompt: {test_prompt}")
    print(f"Prompt tokens: {len(tokenizer.encode(test_prompt, add_special_tokens=False))}")


    # Cell 4
    # Tokenize prompt with chat template
    messages = [
        {"role": "user", "content": test_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Get device - handle device_map="auto" case
    try:
        device = next(ft_model.parameters()).device
        # If device is MPS, switch to CPU to avoid memory issues
        if str(device).startswith('mps'):
            print("WARNING: MPS device detected, switching to CPU to avoid memory limitations")
            device = torch.device("cpu")
            # Move model to CPU if needed
            ft_model = ft_model.to(device)
    except:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"Device: {device}")

    # Cell 5
    # Generate map
    print("Generating map...")

    # Extract input_ids and attention_mask explicitly
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask', None)

    # Store input length for later
    input_length = input_ids.shape[1]

    # Clear cache before generation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        with torch.no_grad():
            # Generate with explicit parameters
            outputs = ft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=6000,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                use_cache=True  # Enable KV cache for memory efficiency
            )
    
        # Decode only the generated tokens (exclude input prompt)
        generated_ids = outputs[0][input_length:]
        generated_content = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
        print(f"Generated tokens: {len(generated_ids)}")
        print(f"Generated length: {len(generated_content)} characters")
        
        # Print raw assistant response
        print("\n" + "="*70)
        print("RAW ASSISTANT RESPONSE")
        print("="*70)
        print(generated_content)
        print("="*70 + "\n")
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("ERROR: Out of memory during generation")
            print("Try reducing max_new_tokens or using a smaller model")
            raise
        else:
            raise

    # Cell 6
    # Extract JSON from generated content
    map_json = extract_json_from_text(generated_content)

    if map_json:
        print("JSON extracted successfully")
        print(f"Keys: {list(map_json.keys())}")
    else:
        print("Failed to extract JSON")


    # Cell 7
    # Validate and reconstruct map
    if map_json:
        print("="*70)
        print("MAP VALIDATION")
        print("="*70)
    
        # Check required fields
        required_fields = ['walkable_tiles', 'player_spawn', 'stairs_spawn', 'width', 'height']
        missing_fields = [f for f in required_fields if f not in map_json]
    
        if missing_fields:
            print(f"WARNING: Missing fields: {missing_fields}")
        else:
            print("All required fields present")
    
        # Get data
        walkable_coords = map_json.get('walkable_tiles', [])
        player_spawn = map_json.get('player_spawn', [0, 0])
        stairs_spawn = map_json.get('stairs_spawn', [0, 0])
        width = map_json.get('width', 56)  # Default to new size: 56x32
        height = map_json.get('height', 32)
    
        print(f"\nWalkable tiles: {len(walkable_coords)} coordinates")
        print(f"Player spawn: {player_spawn}")
        print(f"Stairs spawn: {stairs_spawn}")
        print(f"Map size: {width}x{height}")
    
        # Validate coordinates
        valid_coords = []
        invalid_coords = []
        for coord in walkable_coords:
            if isinstance(coord, list) and len(coord) == 2:
                x, y = coord
                if 0 <= x < width and 0 <= y < height:
                    valid_coords.append(coord)
                else:
                    invalid_coords.append(coord)
            else:
                invalid_coords.append(coord)
    
        print(f"\nValid coordinates: {len(valid_coords)}")
        if invalid_coords:
            print(f"WARNING: Invalid coordinates: {len(invalid_coords)}")
            print(f"   Examples: {invalid_coords[:5]}")
    
        # Reconstruct grid
        print("\nReconstructing grid from coordinates...")
        full_grid = coordinates_to_grid(valid_coords, width=width, height=height)
    
        # Add spawn points (5x5 areas)
        if player_spawn and len(player_spawn) == 2:
            px, py = player_spawn
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if full_grid[ny, nx] == 1:  # Only on floor tiles
                            full_grid[ny, nx] = 2  # Player spawn
    
        if stairs_spawn and len(stairs_spawn) == 2:
            sx, sy = stairs_spawn
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if full_grid[ny, nx] == 1:  # Only on floor tiles
                            full_grid[ny, nx] = 3  # Stairs spawn
    
        print(f"Grid reconstructed: {full_grid.shape}")
        print(f"  Walls (0): {np.sum(full_grid == 0)}")
        print(f"  Floors (1): {np.sum(full_grid == 1)}")
        print(f"  Player spawn (2): {np.sum(full_grid == 2)}")
        print(f"  Stairs spawn (3): {np.sum(full_grid == 3)}")
    
        # Check if spawns are on walkable tiles
        if player_spawn and len(player_spawn) == 2:
            px, py = player_spawn
            if 0 <= px < width and 0 <= py < height:
                if full_grid[py, px] == 0:
                    print(f"WARNING: Player spawn at ({px}, {py}) is on a wall!")
    
        if stairs_spawn and len(stairs_spawn) == 2:
            sx, sy = stairs_spawn
            if 0 <= sx < width and 0 <= sy < height:
                if full_grid[sy, sx] == 0:
                    print(f"WARNING: Stairs spawn at ({sx}, {sy}) is on a wall!")
    
        print("\nMap reconstruction complete!")
    else:
        print("ERROR: Cannot reconstruct map - JSON extraction failed")


    # Cell 8
    # Visualize the map
    if map_json and 'full_grid' in locals():
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for scripts
        import matplotlib.pyplot as plt
    
        # Create color map: 0=black (wall), 1=brown (floor), 2=green (player), 3=red (stairs)
        color_map = np.zeros((height, width, 3), dtype=np.uint8)
        color_map[full_grid == 0] = [0, 0, 0]        # Black walls
        color_map[full_grid == 1] = [139, 69, 19]    # Brown floors
        color_map[full_grid == 2] = [0, 255, 0]      # Green player spawn
        color_map[full_grid == 3] = [255, 0, 0]      # Red stairs spawn
    
        plt.figure(figsize=(10, 10))
        plt.imshow(color_map)
        plt.title(f"Generated Map: {test_prompt}")
        plt.axis('off')
    
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='Wall'),
            Patch(facecolor='#8B4513', label='Floor'),
            Patch(facecolor='green', label='Player Spawn'),
            Patch(facecolor='red', label='Stairs Spawn')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
        plt.tight_layout()
        
        # Save instead of showing
        script_dir = Path(__file__).parent
        viz_path = script_dir / "generated_map_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
    
        print(f"Map visualization saved to {viz_path}")


    # Cell 9
    # Save map for game use and update map_index.json
    if map_json and 'full_grid' in locals():
        # Game expects tiles array with only 0 (wall) and 1 (floor)
        # Spawn points (2, 3) should be converted back to floor tiles (1)
        game_tiles = full_grid.copy()
        game_tiles[game_tiles == 2] = 1  # Convert player spawn areas to floors
        game_tiles[game_tiles == 3] = 1  # Convert stairs spawn areas to floors
        
        # Ensure spawn coordinates are valid single points (not arrays)
        player_spawn_coords = player_spawn if isinstance(player_spawn, list) and len(player_spawn) == 2 else [0, 0]
        stairs_spawn_coords = stairs_spawn if isinstance(stairs_spawn, list) and len(stairs_spawn) == 2 else [0, 0]
        
        # Ensure spawn points are on floor tiles in the final game_tiles array
        px, py = int(player_spawn_coords[0]), int(player_spawn_coords[1])
        sx, sy = int(stairs_spawn_coords[0]), int(stairs_spawn_coords[1])
        
        # Validate and fix player spawn
        if not (0 <= px < width and 0 <= py < height) or game_tiles[py, px] == 0:
            # Find nearest floor tile
            nearest = None
            min_dist = float('inf')
            for y in range(height):
                for x in range(width):
                    if game_tiles[y, x] == 1:  # Floor tile
                        dist = abs(x - px) + abs(y - py)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = [x, y]
            if nearest:
                player_spawn_coords = nearest
                px, py = nearest
                print(f"WARNING: Adjusted player spawn to nearest floor: ({px}, {py})")
        
        # Validate and fix stairs spawn
        if not (0 <= sx < width and 0 <= sy < height) or game_tiles[sy, sx] == 0:
            # Find nearest floor tile
            nearest = None
            min_dist = float('inf')
            for y in range(height):
                for x in range(width):
                    if game_tiles[y, x] == 1:  # Floor tile
                        dist = abs(x - sx) + abs(y - sy)
                        if dist < min_dist:
                            min_dist = dist
                            nearest = [x, y]
            if nearest:
                stairs_spawn_coords = nearest
                sx, sy = nearest
                print(f"WARNING: Adjusted stairs spawn to nearest floor: ({sx}, {sy})")
        
        # Ensure spawn tiles are marked as floors (safety check)
        game_tiles[py, px] = 1  # Player spawn location is floor
        game_tiles[sy, sx] = 1  # Stairs spawn location is floor
        
        # Ensure enemies have x and y properties (not coordinates array)
        game_enemies = []
        for enemy in map_json.get('enemies', []):
            if isinstance(enemy, dict) and 'x' in enemy and 'y' in enemy:
                game_enemies.append({'x': int(enemy['x']), 'y': int(enemy['y'])})
            elif isinstance(enemy, list) and len(enemy) >= 2:
                # If enemy is [x, y] format, convert to {x, y}
                game_enemies.append({'x': int(enemy[0]), 'y': int(enemy[1])})
        
        # Convert to game format (full grid with tiles array)
        # Tiles should be rows (y) then columns (x) - full_grid[y, x] -> tiles[y][x]
        # numpy array: game_tiles.shape = (height, width), accessed as game_tiles[y, x]
        # .tolist() converts to list of rows: [[row0], [row1], ...] where row0 = [col0, col1, ...]
        # This gives us tiles[y][x] which matches game.js: currentMap.tiles[y][x]
        game_map = {
            'tiles': game_tiles.tolist(),  # List of rows (y), each row is columns (x)
            'player_spawn': [px, py],  # [x, y] format as game expects
            'stairs_spawn': [sx, sy],  # [x, y] format as game expects
            'width': int(width),
            'height': int(height),
            'difficulty': map_json.get('difficulty', 'medium'),
            'enemies': game_enemies
        }
    
        # Save to file - use absolute path from script location
        script_dir = Path(__file__).parent
        maps_dir = script_dir.parent.parent / "web_game" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
    
        # Generate unique map ID with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_id = f"map_generated_{timestamp}"
        output_path = maps_dir / f"{map_id}.json"
    
        with open(output_path, 'w') as f:
            json.dump(game_map, f, indent=2)
    
        # Update map_index.json
        index_path = maps_dir / "map_index.json"
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
        
        # Add new map ID if not already present
        if map_id not in map_index['map_ids']:
            map_index['map_ids'].append(map_id)
            map_index['total_maps'] = len(map_index['map_ids'])
        
        # Save updated index
        with open(index_path, 'w') as f:
            json.dump(map_index, f, indent=2)
        
        print(f"\n{'='*70}")
        print("MAP SAVED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Map ID: {map_id}")
        print(f"File: {output_path}")
        print(f"Map Index updated: {len(map_index['map_ids'])} maps total")
        print(f"\nYou can now start the web game and play this map!")
        print(f"  Maps directory: {maps_dir}")
        print(f"  Index file: {index_path}")
        print(f"{'='*70}")
    else:
        print("WARNING: Skipping save - map reconstruction failed")


    # Cell 10
    # Print summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Prompt: {test_prompt}")
    print(f"Generated tokens: {len(generated_ids) if 'generated_ids' in locals() else 'N/A'}")
    print(f"JSON extracted: {'Yes' if map_json else 'No'}")
    print(f"Walkable coordinates: {len(walkable_coords) if map_json else 'N/A'}")
    print(f"Grid reconstructed: {'Yes' if 'full_grid' in locals() else 'No'}")
    print(f"Map saved: {'Yes' if 'output_path' in locals() else 'No'}")
    print("="*70)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a map using the fine-tuned model')
    parser.add_argument('prompt', nargs='?', default=None,
                       help='Prompt for map generation (default: "Generate a medium difficulty dungeon with 6 rooms")')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving the map to web_game/maps/')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Skip generating visualization image')
    
    args = parser.parse_args()
    
    # Pass prompt to main (we'll need to modify main to handle no-save and no-visualization)
    main(prompt=args.prompt)