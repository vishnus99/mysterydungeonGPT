"""
Common functions for converting maps to JSON format.
Used by convert_maps_to_json.py
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def extract_spawn_points_from_array(map_data_list: List[List[int]]) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Extract spawn points from a 2D map array and convert spawn tiles to floor.
    
    Args:
        map_data_list: 2D list representing the map (0=wall, 1=floor, 2=player, 3=stairs)
    
    Returns:
        Tuple of (player_spawn, stairs_spawn) as [x, y] coordinates, or (None, None) if not found
    """
    player_spawn = None
    stairs_spawn = None
    
    for y in range(len(map_data_list)):
        for x in range(len(map_data_list[0])):
            if map_data_list[y][x] == 2:
                player_spawn = [x, y]
                map_data_list[y][x] = 1  # Convert to floor
            elif map_data_list[y][x] == 3:
                stairs_spawn = [x, y]
                map_data_list[y][x] = 1  # Convert to floor
    
    return player_spawn, stairs_spawn


def create_map_json_structure(
    map_id: str,
    map_data_list: List[List[int]],
    player_spawn: Optional[List[int]] = None,
    stairs_spawn: Optional[List[int]] = None,
    fallback_spawns: Optional[Dict[str, int]] = None
) -> Dict:
    """
    Create the JSON structure for a map.
    
    Args:
        map_id: Unique identifier for the map
        map_data_list: 2D list of tiles (0=wall, 1=floor)
        player_spawn: [x, y] player spawn coordinates (or None to use fallback)
        stairs_spawn: [x, y] stairs spawn coordinates (or None to use fallback)
        fallback_spawns: Dict with keys 'player_spawn_x', 'player_spawn_y', 
                        'stairs_spawn_x', 'stairs_spawn_y' for fallback values
    
    Returns:
        Dictionary with map information in JSON format
    """
    # Use fallback coordinates if spawn points not found in array
    if player_spawn is None:
        if fallback_spawns:
            px = fallback_spawns.get('player_spawn_x', -1)
            py = fallback_spawns.get('player_spawn_y', -1)
            if px >= 0 and py >= 0:
                player_spawn = [px, py]
        
        if player_spawn is None:
            player_spawn = [0, 0]  # Default fallback
    
    if stairs_spawn is None:
        if fallback_spawns:
            sx = fallback_spawns.get('stairs_spawn_x', -1)
            sy = fallback_spawns.get('stairs_spawn_y', -1)
            if sx >= 0 and sy >= 0:
                stairs_spawn = [sx, sy]
        
        if stairs_spawn is None:
            stairs_spawn = [0, 0]  # Default fallback
    
    return {
        'map_id': map_id,
        'width': len(map_data_list[0]),
        'height': len(map_data_list),
        'tiles': map_data_list,
        'player_spawn': player_spawn,
        'stairs_spawn': stairs_spawn
    }


def save_map_json(map_info: Dict, output_dir: Path) -> Path:
    """
    Save a map's JSON structure to a file.
    
    Args:
        map_info: Dictionary with map information (must contain 'map_id')
        output_dir: Directory to save the JSON file
    
    Returns:
        Path to the saved file
    """
    output_file = output_dir / f"{map_info['map_id']}.json"
    with open(output_file, 'w') as f:
        json.dump(map_info, f)
    return output_file


def create_map_index(all_map_ids: List[str], output_dir: Path, **extra_metadata) -> Path:
    """
    Create a map_index.json file listing all available maps.
    
    Args:
        all_map_ids: List of map IDs
        output_dir: Directory to save the index file
        **extra_metadata: Additional metadata to include in the index
    
    Returns:
        Path to the created index file
    """
    index = {
        'total_maps': len(all_map_ids),
        'map_ids': all_map_ids,
        **extra_metadata
    }
    
    index_file = output_dir / 'map_index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    return index_file

