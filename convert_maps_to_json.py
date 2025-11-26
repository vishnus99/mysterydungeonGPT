"""
Load maps from HuggingFace dataset and convert them to JSON for the browser game.
"""
import numpy as np
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import logging
from map_converter_common import (
    extract_spawn_points_from_array,
    create_map_json_structure,
    save_map_json,
    create_map_index
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_map_array(map_array_data):
    """
    Parse map_array from HuggingFace dataset.
    It could be stored as numpy array, string, bytes, or need extraction from image.
    """
    if isinstance(map_array_data, np.ndarray):
        return map_array_data
    elif isinstance(map_array_data, (str, bytes)):
        # Try to decode if it's bytes
        if isinstance(map_array_data, bytes):
            try:
                # Try loading as numpy array from bytes (assuming standard shape)
                # Note: HuggingFace may store numpy arrays in various formats
                # If bytes, it's likely a serialized format - try common approaches
                try:
                    # Try as pickle (common for numpy arrays)
                    import pickle
                    return pickle.loads(map_array_data)
                except:
                    # If pickle fails, fall back to image extraction
                    # (bytes format is complex and image extraction is more reliable)
                    return None
            except:
                # If that fails, return None to fall back to image extraction
                return None
        else:
            # String representation
            try:
                return np.array(eval(map_array_data))
            except:
                return None
    return None


def extract_map_from_image(image_pil: Image.Image) -> np.ndarray:
    """Extract map array from PIL Image (fallback method)"""
    img_array = np.array(image_pil)
    
    # If it's RGB (3 channels), check for spawn points
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        height, width = img_array.shape[:2]
        map_array = np.zeros((height, width), dtype=np.uint8)
        
        # Green = player spawn (2), Red = stairs spawn (3)
        # Brown = floor (1), Black = wall (0)
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x]
                if r == 0 and g == 0 and b == 0:
                    map_array[y, x] = 0  # Wall
                elif r == 139 and g == 69 and b == 19:
                    map_array[y, x] = 1  # Floor (brown)
                elif r == 0 and g == 255 and b == 0:
                    map_array[y, x] = 2  # Player spawn (green)
                elif r == 255 and g == 0 and b == 0:
                    map_array[y, x] = 3  # Stairs spawn (red)
                else:
                    map_array[y, x] = 1 if (r + g + b) > 0 else 0
    elif len(img_array.shape) == 2:
        # Grayscale image
        normalized = img_array.astype(np.float32) / 255.0
        map_array = (normalized > 0.5).astype(np.uint8)
    else:
        # Convert to grayscale
        gray = np.mean(img_array, axis=2)
        normalized = gray.astype(np.float32) / 255.0
        map_array = (normalized > 0.5).astype(np.uint8)
    
    return map_array


def convert_map_to_json(map_data, output_dir):
    """
    Convert a single map from HuggingFace dataset to JSON format.
    
    Args:
        map_data: Dictionary containing map data from HuggingFace dataset
        output_dir: Directory to save JSON file
    """
    # Try to get map_array
    map_array = None
    
    # Method 1: Try to parse stored map_array
    if 'map_array' in map_data and map_data['map_array'] is not None:
        map_array = parse_map_array(map_data['map_array'])
    
    # Method 2: Extract from image if map_array parsing failed
    if map_array is None and 'image' in map_data:
        logger.warning(f"Could not parse map_array for {map_data.get('map_id', 'unknown')}, extracting from image")
        map_array = extract_map_from_image(map_data['image'])
    
    if map_array is None:
        raise ValueError(f"Could not extract map_array for map {map_data.get('map_id', 'unknown')}")
    
    # Convert to list format
    map_data_list = map_array.tolist()
    
    # Extract spawn points from map_array
    player_spawn, stairs_spawn = extract_spawn_points_from_array(map_data_list)
    
    # Prepare fallback coordinates from stored metadata
    fallback_spawns = {
        'player_spawn_x': map_data.get('player_spawn_x', -1),
        'player_spawn_y': map_data.get('player_spawn_y', -1),
        'stairs_spawn_x': map_data.get('stairs_spawn_x', -1),
        'stairs_spawn_y': map_data.get('stairs_spawn_y', -1)
    }
    
    # Create JSON structure
    map_info = create_map_json_structure(
        map_id=map_data.get('map_id', 'unknown'),
        map_data_list=map_data_list,
        player_spawn=player_spawn,
        stairs_spawn=stairs_spawn,
        fallback_spawns=fallback_spawns
    )
    
    # Save to file
    save_map_json(map_info, Path(output_dir))
    
    return map_info


def convert_maps_to_json(hf_repo_id: str, output_dir: str, split: str = 'train', max_maps: int = None):
    """
    Load maps from HuggingFace dataset and convert to JSON format for browser game.
    
    Args:
        hf_repo_id: HuggingFace dataset repository ID (e.g., 'vishnusm/mysterydungeonmaps')
        output_dir: Directory to save JSON maps
        split: Dataset split to load ('train', 'val', etc.)
        max_maps: Maximum number of maps to convert (None for all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading dataset from HuggingFace: {hf_repo_id}")
    logger.info(f"Split: {split}")
    
    # Load dataset
    try:
        dataset = load_dataset(hf_repo_id, split=split)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    logger.info(f"Loaded {len(dataset)} maps from dataset")
    
    # Limit number of maps if specified
    if max_maps is not None:
        dataset = dataset.select(range(min(max_maps, len(dataset))))
        logger.info(f"Processing {len(dataset)} maps (limited from original)")
    
    # Convert each map
    all_maps = []
    failed_count = 0
    
    for i, map_data in enumerate(dataset):
        try:
            map_info = convert_map_to_json(map_data, output_path)
            all_maps.append(map_info['map_id'])
            
            if (i + 1) % 100 == 0:
                logger.info(f"Converted {i + 1}/{len(dataset)} maps...")
        except Exception as e:
            logger.error(f"Error converting map {i}: {e}")
            failed_count += 1
            continue
    
    # Create index file
    index_file = create_map_index(
        all_maps,
        output_path,
        source=hf_repo_id,
        split=split
    )
    
    logger.info(f"\nConversion complete!")
    logger.info(f"  Successfully converted: {len(all_maps)} maps")
    logger.info(f"  Failed: {failed_count} maps")
    logger.info(f"  Output directory: {output_path.absolute()}")
    logger.info(f"  Index file: {index_file.absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load maps from HuggingFace and convert to JSON")
    parser.add_argument(
        '--repo-id',
        type=str,
        default='vishnusm/mysterydungeonmaps',
        help='HuggingFace dataset repository ID'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./web_game/maps',
        help='Output directory for JSON maps'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to load (train, val, etc.)'
    )
    parser.add_argument(
        '--max-maps',
        type=int,
        default=None,
        help='Maximum number of maps to convert (None for all)'
    )
    
    args = parser.parse_args()
    
    convert_maps_to_json(
        hf_repo_id=args.repo_id,
        output_dir=args.output_dir,
        split=args.split,
        max_maps=args.max_maps
    )

