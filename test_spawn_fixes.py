#!/usr/bin/env python3
"""
Test script to verify spawn point fixes
Tests existing maps from HuggingFace dataset
"""

import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataset.mapgenerator import is_map_playable
from mysterydungeonGPT.helpers import format_map_for_training, coordinates_to_grid
import json

def test_spawn_fixes(dataset_name="teamgas/mysterydungeondata", num_maps=50):
    """Test spawn point validation on existing maps from HuggingFace"""
    
    print("="*70)
    print("TESTING SPAWN POINT VALIDATION ON EXISTING MAPS")
    print("="*70)
    print(f"Loading dataset: {dataset_name}")
    print(f"Testing {num_maps} maps...\n")
    
    # Load dataset
    dataset = load_dataset(dataset_name)
    df = dataset['train']
    
    # Limit to num_maps
    test_indices = list(range(min(num_maps, len(df))))
    
    print(f"Loaded {len(df)} total maps")
    print(f"Testing maps: {test_indices[0]} to {test_indices[-1]}\n")
    print("="*70)
    print("VALIDATING SPAWN POINTS")
    print("="*70)
    
    # Validate each map
    valid_count = 0
    invalid_count = 0
    issues = []
    fixed_count = 0
    
    for i in test_indices:
        example = df[i]
        
        # Get spawn coordinates from metadata
        player_spawn_x = example.get('player_spawn_x', -1)
        player_spawn_y = example.get('player_spawn_y', -1)
        stairs_spawn_x = example.get('stairs_spawn_x', -1)
        stairs_spawn_y = example.get('stairs_spawn_y', -1)
        
        player_spawn = (player_spawn_x, player_spawn_y) if player_spawn_x != -1 else (-1, -1)
        stairs_spawn = (stairs_spawn_x, stairs_spawn_y) if stairs_spawn_x != -1 else (-1, -1)
        
        # Get map_array from dataset - stored as JSON string
        map_array_raw = example.get('map_array')
        
        if map_array_raw is None:
            print(f"WARNING: Map {i} has no map_array, skipping")
            continue
        
        # Parse JSON string to numpy array
        try:
            parsed = json.loads(map_array_raw)
            map_array = np.array(parsed, dtype=np.uint8)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"ERROR: Map {i} - Failed to parse map_array JSON: {e}")
            continue
        
        # Get map dimensions from dataset
        map_width = example.get('width', 56)
        map_height = example.get('height', 32)
        
        # Validate shape
        if map_array.shape != (map_height, map_width):
            print(f"WARNING: Map {i} has invalid map_array shape: {map_array.shape}, expected ({map_height}, {map_width})")
            continue
        
        # Parse enemies
        enemies_list = []
        if 'enemies' in example:
            try:
                enemies_list = json.loads(example['enemies'])
            except:
                enemies_list = []
        
        # Check if spawns are on floor tiles in original map_array
        player_valid = False
        stairs_valid = False
        player_issue = None
        stairs_issue = None
        
        if player_spawn[0] != -1 and player_spawn[1] != -1:
            px, py = player_spawn
            if 0 <= px < map_width and 0 <= py < map_height:
                tile_value = map_array[py, px]
                player_valid = tile_value in [1, 2]
                if not player_valid:
                    player_issue = f"Player spawn ({px}, {py}) is on tile value {tile_value}, not floor"
            else:
                player_issue = f"Player spawn ({px}, {py}) is out of bounds"
        
        if stairs_spawn[0] != -1 and stairs_spawn[1] != -1:
            sx, sy = stairs_spawn
            if 0 <= sx < map_width and 0 <= sy < map_height:
                tile_value = map_array[sy, sx]
                stairs_valid = tile_value in [1, 3]
                if not stairs_valid:
                    stairs_issue = f"Stairs spawn ({sx}, {sy}) is on tile value {tile_value}, not floor"
            else:
                stairs_issue = f"Stairs spawn ({sx}, {sy}) is out of bounds"
        
        # Test if helpers.py fix would work
        would_be_fixed = False
        if (player_issue or stairs_issue):
            # Test the fix logic from helpers.py
            try:
                prompt, json_output = format_map_for_training(example)
                json_data = json.loads(json_output)
                
                # Check if spawns are now in walkable_coords
                walkable_coords = json_data.get('walkable_tiles', [])
                walkable_set = set(tuple(c) for c in walkable_coords)
                
                fixed_player = tuple(json_data.get('player_spawn', [])) in walkable_set
                fixed_stairs = tuple(json_data.get('stairs_spawn', [])) in walkable_set
                
                if fixed_player and fixed_stairs:
                    would_be_fixed = True
                    fixed_count += 1
            except Exception as e:
                pass  # Skip if format_map_for_training fails
        
        # Check playability
        # For BFS, spawn points need to be on walkable tiles (value != 0)
        # If spawn coordinates are on walls, that's the actual problem
        bfs_map = map_array.copy()
        
        # Ensure spawn points are marked as walkable for BFS
        # (BFS checks map_array[y][x] != 0, so 1, 2, or 3 all work)
        if player_spawn[0] != -1 and player_spawn[1] != -1:
            px, py = player_spawn
            if 0 <= px < map_width and 0 <= py < map_height:
                if bfs_map[py, px] == 0:
                    # Spawn is on a wall - this is the problem!
                    # For BFS to work, we need to make it walkable, but this indicates a real issue
                    bfs_map[py, px] = 1  # Temporarily make walkable for BFS check
                else:
                    bfs_map[py, px] = 2  # Mark as player spawn
        if stairs_spawn[0] != -1 and stairs_spawn[1] != -1:
            sx, sy = stairs_spawn
            if 0 <= sx < map_width and 0 <= sy < map_height:
                if bfs_map[sy, sx] == 0:
                    # Spawn is on a wall - this is the problem!
                    bfs_map[sy, sx] = 1  # Temporarily make walkable for BFS check
                else:
                    bfs_map[sy, sx] = 3  # Mark as stairs spawn
        
        is_playable = is_map_playable(bfs_map, enemies_list, player_spawn, stairs_spawn)
        
        if player_valid and stairs_valid and is_playable:
            valid_count += 1
            if i < 10:  # Only print first 10 to avoid spam
                print(f"Map {i}: ✓ Valid (Player: {player_spawn}, Stairs: {stairs_spawn})")
        else:
            invalid_count += 1
            problems = []
            if player_issue:
                problems.append("Player spawn not on floor")
                issues.append(f"Map {i}: {player_issue}")
            if stairs_issue:
                problems.append("Stairs spawn not on floor")
                issues.append(f"Map {i}: {stairs_issue}")
            if not is_playable:
                problems.append("Not playable (BFS failed)")
                # Debug BFS failure
                if i < 5:  # Only debug first 5
                    px, py = player_spawn
                    sx, sy = stairs_spawn
                    player_tile = bfs_map[py, px] if (0<=px<32 and 0<=py<32) else 'OOB'
                    stairs_tile = bfs_map[sy, sx] if (0<=sx<32 and 0<=sy<32) else 'OOB'
                    print(f"  DEBUG BFS: Player ({px}, {py}) tile={player_tile}, "
                          f"Stairs ({sx}, {sy}) tile={stairs_tile}")
                    print(f"  DEBUG: Map shape={bfs_map.shape}, Floors={np.sum(bfs_map == 1)}, "
                          f"Player spawns={np.sum(bfs_map == 2)}, Stairs spawns={np.sum(bfs_map == 3)}")
            
            fix_status = " (would be fixed by helpers.py)" if would_be_fixed else ""
            if i < 10:  # Only print first 10 to avoid spam
                print(f"Map {i}: ✗ Invalid - {', '.join(problems)}{fix_status}")
                print(f"  Player: {player_spawn}, Stairs: {stairs_spawn}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total maps tested: {len(test_indices)}")
    print(f"Valid maps: {valid_count}")
    print(f"Invalid maps: {invalid_count}")
    print(f"Maps that would be fixed by helpers.py: {fixed_count}")
    print(f"Success rate: {valid_count/len(test_indices)*100:.1f}%")
    print(f"Fixable rate: {fixed_count/invalid_count*100:.1f}%" if invalid_count > 0 else "Fixable rate: N/A")
    
    if issues:
        print(f"\nIssues found ({len(issues)} total):")
        # Show first 10 issues
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
    else:
        print("\n✓ All spawn points are valid!")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    if invalid_count > 0:
        print(f"- {invalid_count} maps have invalid spawn points")
        print(f"- {fixed_count} of these would be fixed by helpers.py validation")
        print(f"- Consider regenerating the dataset to apply the fixes at generation time")
        print(f"- Or use the helpers.py validation during training (already implemented)")
    else:
        print("✓ All tested maps have valid spawn points!")
    
    return valid_count == len(test_indices)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test spawn point validation on existing maps")
    parser.add_argument("--dataset", default="teamgas/mysterydungeondata",
                       help="HuggingFace dataset name (default: teamgas/mysterydungeondata)")
    parser.add_argument("--num-maps", type=int, default=50,
                       help="Number of maps to test (default: 50)")
    
    args = parser.parse_args()
    
    success = test_spawn_fixes(args.dataset, args.num_maps)
    sys.exit(0 if success else 1)

