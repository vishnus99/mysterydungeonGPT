#!/usr/bin/env python3
"""
Verify Training Data Format
Shows the prompt and JSON format used for training, and visualizes the reconstructed map
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mysterydungeonGPT.helpers import format_map_for_training, coordinates_to_grid

def verify_training_data(dataset_name="teamgas/mysterydungeondata", indices=None):
    """Verify training data format for given indices"""
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    df = dataset['train']
    
    if indices is None:
        indices = [0, 1, 2]  # Default to first 3 maps
    
    for idx in indices:
        print("\n" + "="*70)
        print(f"MAP {idx}")
        print("="*70)
        
        # Get raw example
        example = df[idx]
        
        # Format for training
        prompt, json_output = format_map_for_training(example)
        
        # Parse JSON
        json_data = json.loads(json_output)
        
        # Print prompt
        print("\nPROMPT:")
        print("-" * 70)
        print(prompt)
        
        # Print JSON (formatted)
        print("\nJSON OUTPUT:")
        print("-" * 70)
        print(json.dumps(json_data, indent=2))
        
        # Print JSON stats
        print("\nJSON STATISTICS:")
        print("-" * 70)
        print(f"Walkable tiles: {len(json_data.get('walkable_tiles', []))}")
        print(f"Player spawn: {json_data.get('player_spawn')}")
        print(f"Stairs spawn: {json_data.get('stairs_spawn')}")
        print(f"Width: {json_data.get('width')}")
        print(f"Height: {json_data.get('height')}")
        print(f"Difficulty: {json_data.get('difficulty')}")
        print(f"Enemies: {len(json_data.get('enemies', []))}")
        
        # DEBUG: Check the original map_array before coordinate extraction
        print("\nDEBUG: Checking original map data...")
        print("-" * 70)
        
        # Extract map_array directly to see what we're working with
        map_array_raw = example.get('map_array')
        map_width = example.get('width', 56)
        map_height = example.get('height', 32)
        
        # Parse map_array from JSON string
        if map_array_raw is None:
            print(f"ERROR: Map {i} has no map_array")
            continue
        
        if isinstance(map_array_raw, str):
            # Parse JSON string
            try:
                parsed = json.loads(map_array_raw)
                map_array_raw = np.array(parsed, dtype=np.uint8)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"ERROR: Map {i} - Failed to parse map_array JSON: {e}")
                continue
        elif isinstance(map_array_raw, list):
            map_array_raw = np.array(map_array_raw, dtype=np.uint8)
        elif not isinstance(map_array_raw, np.ndarray):
            map_array_raw = np.array(map_array_raw, dtype=np.uint8)
        
        # Validate shape
        if map_array_raw.shape != (map_height, map_width):
            print(f"WARNING: Map {i} has invalid map_array shape: {map_array_raw.shape}, expected ({map_height}, {map_width})")
            continue
        
        print(f"Map array shape: {map_array_raw.shape}")
        print(f"Map array dtype: {map_array_raw.dtype}")
        print(f"Unique values in map_array: {np.unique(map_array_raw)}")
        print(f"Number of floor tiles (1): {np.sum(map_array_raw == 1)}")
        print(f"Number of wall tiles (0): {np.sum(map_array_raw == 0)}")
        print(f"Number of player spawn (2): {np.sum(map_array_raw == 2)}")
        print(f"Number of stairs spawn (3): {np.sum(map_array_raw == 3)}")
        
        # Reconstruct grid from coordinates
        walkable_coords = json_data.get('walkable_tiles', [])
        width = json_data.get('width', 56)
        height = json_data.get('height', 32)
        
        # DEBUG: Check coordinates extracted
        print(f"\nDEBUG: Coordinate extraction check:")
        print(f"  Coordinates extracted: {len(walkable_coords)}")
        if len(walkable_coords) > 0:
            print(f"  First 5 coordinates: {walkable_coords[:5]}")
            print(f"  Last 5 coordinates: {walkable_coords[-5:]}")
            
            # Check coordinate ranges
            xs = [c[0] for c in walkable_coords]
            ys = [c[1] for c in walkable_coords]
            print(f"  X range: {min(xs)} to {max(xs)} (width: {width})")
            print(f"  Y range: {min(ys)} to {max(ys)} (height: {height})")
        
        grid = coordinates_to_grid(walkable_coords, width=width, height=height)
        
        # DEBUG: Verify coordinate reconstruction
        print(f"\nDEBUG: Coordinate reconstruction check:")
        print(f"  Input coordinates: {len(walkable_coords)}")
        print(f"  Grid floors after reconstruction: {np.sum(grid == 1)}")
        print(f"  Grid walls after reconstruction: {np.sum(grid == 0)}")
        if len(walkable_coords) > 0 and np.sum(grid == 1) == 0:
            print("  ERROR: No floors in reconstructed grid!")
            print(f"  Sample coordinates: {walkable_coords[:10]}")
            # Try manual reconstruction
            test_grid = np.zeros((height, width), dtype=np.uint8)
            for coord in walkable_coords[:10]:  # Just first 10
                x, y = coord
                print(f"    Setting grid[{y}, {x}] = 1")
                if 0 <= x < width and 0 <= y < height:
                    test_grid[y, x] = 1
            print(f"  Test grid floors (first 10 coords): {np.sum(test_grid == 1)}")
        
        # Add spawn points with validation
        player_spawn = json_data.get('player_spawn', [0, 0])
        stairs_spawn = json_data.get('stairs_spawn', [0, 0])
        
        print(f"\nDEBUG: Spawn point validation:")
        print("-" * 70)
        
        # Check if spawn points are in walkable_coords
        walkable_coords_set = set(tuple(c) for c in walkable_coords)
        
        if player_spawn and len(player_spawn) == 2:
            px, py = player_spawn
            player_coord = (px, py)
            is_walkable = player_coord in walkable_coords_set
            is_in_bounds = 0 <= px < width and 0 <= py < height
            is_floor = is_in_bounds and grid[py, px] == 1
            
            print(f"Player spawn: {player_spawn}")
            print(f"  In bounds: {is_in_bounds}")
            print(f"  In walkable_coords: {is_walkable}")
            print(f"  On floor tile: {is_floor}")
            
            if is_in_bounds:
                if not is_floor:
                    print(f"  WARNING: Player spawn is NOT on a floor tile!")
                grid[py, px] = 2  # Player spawn
            else:
                print(f"  ERROR: Player spawn is out of bounds!")
        
        if stairs_spawn and len(stairs_spawn) == 2:
            sx, sy = stairs_spawn
            stairs_coord = (sx, sy)
            is_walkable = stairs_coord in walkable_coords_set
            is_in_bounds = 0 <= sx < width and 0 <= sy < height
            is_floor = is_in_bounds and grid[sy, sx] == 1
            
            print(f"Stairs spawn: {stairs_spawn}")
            print(f"  In bounds: {is_in_bounds}")
            print(f"  In walkable_coords: {is_walkable}")
            print(f"  On floor tile: {is_floor}")
            
            if is_in_bounds:
                if not is_floor:
                    print(f"  WARNING: Stairs spawn is NOT on a floor tile!")
                grid[sy, sx] = 3  # Stairs spawn
            else:
                print(f"  ERROR: Stairs spawn is out of bounds!")
        
        # Check BFS reachability
        print(f"\nDEBUG: BFS reachability check:")
        print("-" * 70)
        
        # Import BFS function
        from src.playablemap import playablebfs
        
        # Create a copy of grid for BFS (needs spawn points as 2 and 3)
        bfs_grid = grid.copy()
        path = playablebfs(bfs_grid)
        
        if path:
            print(f"  BFS path found: {len(path)} steps")
            print(f"  Path is valid: YES")
        else:
            print(f"  BFS path found: NO")
            print(f"  Path is valid: NO - SPAWN POINTS ARE NOT REACHABLE!")
            print(f"  This map violates the playability requirement!")
        
        # Visualize
        print("\nRECONSTRUCTED MAP:")
        print("-" * 70)
        print_ascii_map(grid)
        
        # Show matplotlib visualization
        visualize_map(grid, title=f"Map {idx}: {prompt[:50]}...")
        
        print("\n" + "="*70)

def print_ascii_map(grid):
    """Print ASCII representation of map"""
    ascii_chars = {0: '#', 1: '.', 2: 'P', 3: 'S'}
    for row in grid:
        print(''.join([ascii_chars.get(int(val), '?') for val in row]))

def visualize_map(grid, title="Map"):
    """Visualize map with matplotlib"""
    color_map = np.zeros((*grid.shape, 3), dtype=np.uint8)
    color_map[grid == 0] = [0, 0, 0]        # Black walls
    color_map[grid == 1] = [139, 69, 19]    # Brown floors
    color_map[grid == 2] = [0, 255, 0]      # Green player spawn
    color_map[grid == 3] = [255, 0, 0]      # Red stairs spawn
    
    plt.figure(figsize=(10, 10))
    plt.imshow(color_map)
    plt.title(title)
    plt.axis('off')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Wall'),
        Patch(facecolor='#8B4513', label='Floor'),
        Patch(facecolor='green', label='Player Spawn'),
        Patch(facecolor='red', label='Stairs Spawn')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify training data format")
    parser.add_argument("--dataset", default="teamgas/mysterydungeondata", 
                       help="HuggingFace dataset name")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2],
                       help="Map indices to verify (default: 0 1 2)")
    
    args = parser.parse_args()
    
    verify_training_data(args.dataset, args.indices)