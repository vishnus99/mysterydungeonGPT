import os
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from datasets import Dataset, Features, Image as HFImage, Value
from huggingface_hub import HfApi, Repository
import logging
import sys
import shutil

#Logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.RandomGen import RandomGenerator
from src.DungeonAlgorithm import Properties, DungeonData, generate_floor

def is_map_playable(map_array: np.ndarray, enemies: List[Dict], player_spawn: Tuple[int, int], stairs_spawn: Tuple[int, int]) -> bool:
    """
    Check if a map is playable by verifying:
    1. Player spawn to stairs is reachable
    2. Player spawn to each enemy spawn is reachable
    
    Args:
        map_array: The map array (0=wall, 1=floor, 2=player spawn, 3=stairs spawn)
        enemies: List of enemy dictionaries with 'x' and 'y' keys
        player_spawn: (x, y) player spawn position
        stairs_spawn: (x, y) stairs spawn position
    
    Returns:
        True if map is playable (all paths exist), False otherwise
    """
    # Check player spawn to stairs
    if player_spawn[0] == -1 or player_spawn[1] == -1:
        # logger.warning("Invalid player spawn position")
        return False
    
    if stairs_spawn[0] == -1 or stairs_spawn[1] == -1:
        # logger.warning("Invalid stairs spawn position")
        return False
    
    if not is_reachable_from_spawn(map_array, player_spawn, stairs_spawn):
        # logger.debug("Player spawn to stairs is not reachable")
        return False
    
    # Check player spawn to each enemy spawn
    # If there are no enemies, the map is still playable (player can reach stairs)
    if not enemies:
        return True
    
    for enemy in enemies:
        if 'x' not in enemy or 'y' not in enemy:
            # logger.warning(f"Enemy missing x or y coordinates: {enemy}")
            continue
        
        enemy_pos = (enemy['x'], enemy['y'])
        if not is_reachable_from_spawn(map_array, player_spawn, enemy_pos):
            # logger.debug(f"Player spawn to enemy at ({enemy_pos[0]}, {enemy_pos[1]}) is not reachable")
            return False
    
    return True

def is_reachable_from_spawn(map_array: np.ndarray, start_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> bool:
    """
    Check if target position is reachable from start position using BFS with 8-directional movement.
    
    Args:
        map_array: The map array (0=wall, 1=floor, 2=player spawn, 3=stairs spawn)
        start_pos: (x, y) starting position
        target_pos: (x, y) target position
    
    Returns:
        True if target is reachable, False otherwise
    """
    height, width = map_array.shape
    start_x, start_y = start_pos
    target_x, target_y = target_pos
    
    # Validate positions
    if not (0 <= start_x < width and 0 <= start_y < height and
            0 <= target_x < width and 0 <= target_y < height):
        return False
    
    # Check if positions are walkable (not walls)
    if map_array[start_y][start_x] == 0 or map_array[target_y][target_x] == 0:
        return False
    
    if start_pos == target_pos:
        return True
    
    # BFS with 8-directional movement
    from collections import deque
    visited = [[False] * width for _ in range(height)]
    queue = deque([start_pos])
    visited[start_y][start_x] = True
    
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        current_x, current_y = queue.popleft()
        
        if (current_x, current_y) == target_pos:
            return True
        
        for dx, dy in directions:
            new_x, new_y = current_x + dx, current_y + dy
            
            if (0 <= new_x < width and 0 <= new_y < height and
                not visited[new_y][new_x] and
                map_array[new_y][new_x] != 0):  # Walkable (floor or spawn area)
                visited[new_y][new_x] = True
                queue.append((new_x, new_y))
    
    return False




class MysteryDungeonMapGenerator: 
    # Difficulty configurations
    DIFFICULTY_CONFIGS = {
        'easy': {
            'enemy_types': ['basic', 'weak'],  # Only basic enemies
            'enemy_count_range': (2, 5),  # 2-5 enemies
            'enemy_density': 0.02  # 1 enemy per 50 floor tiles
        },
        'medium': {
            'enemy_types': ['basic', 'aggressive', 'patrol'],  # Mix of types
            'enemy_count_range': (5, 10),  # 5-10 enemies
            'enemy_density': 0.03  # 1 enemy per 33 floor tiles
        },
        'hard': {
            'enemy_types': ['aggressive', 'patrol', 'elite', 'boss'],  # Stronger enemies
            'enemy_count_range': (10, 20),  # 10-20 enemies
            'enemy_density': 0.05  # 1 enemy per 20 floor tiles
        }
    }
    
    def __init__(self,
                 output_dir: str = "./maps",
                 image_size: Tuple[int,int] = (256, 256),
                 map_size: Tuple[int,int] = (32, 32),
                 hf_repo_id: str = "teamgas/mysterydungeondata"):

        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.map_size = map_size
        self.hf_repo_id = hf_repo_id

        #Output dirs
        self.images_dir = self.output_dir / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset features
        self.features = Features({
            'image': HFImage(),
            'map_id': Value('string'),
            'width': Value('int64'),
            'height': Value('int64'),
            'complexity': Value('float32'),
            'room_count': Value('int64'),
            'corridor_length': Value('float32'),
            'difficulty': Value('string'),
            'generation_params': Value('string'),
            'hash': Value('string'),
            'map_array': Value('string'),
            'player_spawn_x': Value('int64'),
            'player_spawn_y': Value('int64'),
            'stairs_spawn_x': Value('int64'),
            'stairs_spawn_y': Value('int64'),
            'enemies': Value('string')
        })

    def generate_enemies(self, 
                     map_array: np.ndarray, 
                     player_spawn: Tuple[int, int],
                     stairs_spawn: Tuple[int, int],
                     difficulty: str = 'medium',
                     original_tiles = None) -> List[Dict]:
        """
        Generate enemy spawn positions. Uses ONLY original_tiles as source of truth.
        
        Args:
            map_array: The map array (for reachability check only)
            player_spawn: (x, y) player spawn position
            stairs_spawn: (x, y) stairs spawn position
            difficulty: 'easy', 'medium', or 'hard' (used only for count)
            original_tiles: Original DungeonData.list_tiles for validation (REQUIRED)
        
        Returns:
            List of enemy dictionaries with only 'x' and 'y' fields
        """
        import random
        
        if original_tiles is None:
            # logger.error("original_tiles is REQUIRED for enemy generation")
            return []
        
        height, width = map_array.shape
        player_spawn_x, player_spawn_y = player_spawn
        stairs_spawn_x, stairs_spawn_y = stairs_spawn
        
        # logger.info(f"=== ENEMY GENERATION DEBUG START ===")
        # logger.info(f"map_array shape: {map_array.shape} (height={height}, width={width})")
        # logger.info(f"original_tiles dimensions: {len(original_tiles)}x{len(original_tiles[0])}")
        # logger.info(f"Player spawn: ({player_spawn_x}, {player_spawn_y})")
        # logger.info(f"Stairs spawn: ({stairs_spawn_x}, {stairs_spawn_y})")
        
        # original_tiles is [x][y] where x=0-55, y=0-31
        # Verify dimensions
        if len(original_tiles) != 32 or len(original_tiles[0]) != 32:
            # logger.error(f"original_tiles has wrong dimensions: {len(original_tiles)}x{len(original_tiles[0])}, expected 32x32")
            return []
        
        # Step 1: Find all valid floor tiles using ONLY original_tiles
        floor_positions = []
        floor_tile_count = 0
        wall_tile_count = 0
        agreement_mismatch_count = 0
        spawn_area_excluded_count = 0
        
        for y in range(height):  # y = 0 to 31
            for x in range(width):  # x = 0 to 55
                # original_tiles is indexed as [x][y]
                if x >= len(original_tiles) or y >= len(original_tiles[0]):
                    continue
                
                tile = original_tiles[x][y]
                
                # ONLY check original_tiles - must be a walkable floor tile
                if (tile[0x0] & 0x3) != 1:
                    if (tile[0x0] & 0x3) == 0:  # Wall
                        wall_tile_count += 1
                    continue  # Not a floor tile
                
                floor_tile_count += 1

                # ENFORCE AGREEMENT: map_array must agree with original_tiles
                # If original_tiles says floor, map_array should be 1 (floor), 2 (player spawn), or 3 (stairs spawn)
                # It should NEVER be 0 (wall)
                if map_array[y, x] == 0:
                    # They disagree - original_tiles says floor but map_array says wall
                    agreement_mismatch_count += 1
                    if agreement_mismatch_count <= 5:  # Log first 5 mismatches
                        # logger.error(f"DATA INCONSISTENCY #{agreement_mismatch_count}: Tile ({x}, {y}) - original_tiles says FLOOR (tile[0x0]={tile[0x0]:02x}, &0x3={(tile[0x0] & 0x3)}) but map_array says WALL (value={map_array[y, x]})")
                        pass
                    continue

                # Only accept actual floor tiles (value 1), not spawn areas (2 or 3)
                if map_array[y, x] != 1:
                    continue  # Skip spawn areas

                # Exclude spawn areas (5x5 areas around player and stairs)
                in_player_area = (player_spawn_x != -1 and player_spawn_y != -1 and
                                abs(x - player_spawn_x) <= 2 and abs(y - player_spawn_y) <= 2)
                in_stairs_area = (stairs_spawn_x != -1 and stairs_spawn_y != -1 and
                                abs(x - stairs_spawn_x) <= 2 and abs(y - stairs_spawn_y) <= 2)

                if in_player_area or in_stairs_area:
                    spawn_area_excluded_count += 1
                    continue

                floor_positions.append((x, y))
        
        # logger.info(f"=== STEP 1: Floor Tile Analysis ===")
        # logger.info(f"  Total floor tiles (original_tiles): {floor_tile_count}")
        # logger.info(f"  Total wall tiles (original_tiles): {wall_tile_count}")
        # logger.info(f"  Agreement mismatches (original_tiles=floor, map_array=wall): {agreement_mismatch_count}")
        # logger.info(f"  Spawn area exclusions: {spawn_area_excluded_count}")
        # logger.info(f"  Valid floor positions after Step 1: {len(floor_positions)}")
        
        # if len(floor_positions) > 0 and len(floor_positions) <= 20:
        #     logger.info(f"  Sample floor positions: {floor_positions[:10]}")
        
        # Step 2: Filter to only reachable positions from player spawn
        if player_spawn_x != -1 and player_spawn_y != -1:
            reachable_count = 0
            unreachable_count = 0
            reachable_positions = []
            for pos in floor_positions:
                if is_reachable_from_spawn(map_array, player_spawn, pos):
                    reachable_positions.append(pos)
                    reachable_count += 1
                else:
                    unreachable_count += 1
                    if unreachable_count <= 5:  # Log first 5 unreachable
                        # logger.debug(f"  Unreachable position: {pos}")
                        pass
            floor_positions = reachable_positions
            # logger.info(f"=== STEP 2: Reachability Check ===")
            # logger.info(f"  Reachable positions: {reachable_count}")
            # logger.info(f"  Unreachable positions: {unreachable_count}")
            # logger.info(f"  Valid floor positions after Step 2: {len(floor_positions)}")
        else:
            # logger.warning("Player spawn is invalid, skipping reachability check")
            pass
        
        if len(floor_positions) == 0:
            # logger.warning(f"=== RESULT: No valid floor positions found for enemy placement ===")
            # logger.warning(f"  This map cannot spawn enemies!")
            return []
        
        # Step 3: Determine enemy count based on difficulty
        config = self.DIFFICULTY_CONFIGS.get(difficulty, self.DIFFICULTY_CONFIGS['medium'])
        min_count, max_count = config['enemy_count_range']
        num_enemies = min(random.randint(min_count, max_count), len(floor_positions))
        
        # logger.info(f"=== STEP 3: Enemy Count Selection ===")
        # logger.info(f"  Difficulty: {difficulty}")
        # logger.info(f"  Count range: {min_count}-{max_count}")
        # logger.info(f"  Selected count: {num_enemies}")
        
        if num_enemies <= 0:
            # logger.warning(f"=== RESULT: No enemies to spawn (count={num_enemies}) ===")
            return []
        
        # Step 4: Randomly select positions
        selected_positions = random.sample(floor_positions, num_enemies)
        # logger.info(f"=== STEP 4: Position Selection ===")
        # logger.info(f"  Selected {len(selected_positions)} positions: {selected_positions}")
        
        # Step 5: Create simple enemy objects with FINAL validation using original_tiles
        enemies = []
        validation_failed = []
        for x, y in selected_positions:
            # Final validation: double-check with original_tiles
            if x < len(original_tiles) and y < len(original_tiles[0]):
                tile = original_tiles[x][y]
                tile_value = tile[0x0]
                tile_walkable = (tile_value & 0x3) == 1
                map_value = map_array[y, x] if y < map_array.shape[0] and x < map_array.shape[1] else -1
                
                if tile_walkable:  # Must be floor
                    enemies.append({'x': int(x), 'y': int(y)})
                    # logger.debug(f"  ✓ Enemy at ({x}, {y}): original_tiles[0x0]={tile_value:02x} (walkable), map_array={map_value}")
                else:
                    validation_failed.append((x, y, f"not floor (tile[0x0]={tile_value:02x}, &0x3={tile_value & 0x3})"))
                    # logger.warning(f"  ✗ Enemy at ({x}, {y}) failed final validation - {validation_failed[-1][2]}")
            else:
                validation_failed.append((x, y, f"out of bounds (x={x} >= {len(original_tiles)} or y={y} >= {len(original_tiles[0])})"))
                # logger.warning(f"  ✗ Enemy at ({x}, {y}) is out of bounds")
        
        # logger.info(f"=== STEP 5: Final Validation ===")
        # logger.info(f"  Passed validation: {len(enemies)}")
        # logger.info(f"  Failed validation: {len(validation_failed)}")
        # if validation_failed:
        #     logger.warning(f"  Failed positions: {validation_failed}")
        
        # if len(enemies) < len(selected_positions):
        #     logger.warning(f"Removed {len(selected_positions) - len(enemies)} enemies that failed final validation")
        
        # logger.info(f"=== ENEMY GENERATION DEBUG END ===")
        # logger.info(f"Final enemy count: {len(enemies)}")
        # if enemies:
        #     logger.info(f"Enemy positions: {enemies}")
        
        return enemies
    def generate_map(self,
                     layout_type: int = 1,
                     room_count: int = 5,
                     complexity: float = 0.5,
                     difficulty: str = 'medium') -> Tuple[np.ndarray, Dict]:
        
        # Validate difficulty
        if difficulty not in self.DIFFICULTY_CONFIGS:
            difficulty = 'medium'  # Default fallback
        
        maximum_connectivity = 15
        Properties.nb_rooms = room_count
        Properties.layout = layout_type
        Properties.floor_connectivity = int(complexity * maximum_connectivity)

        DungeonData.clear_tiles()

        generate_floor()

        player_spawn_x = DungeonData.player_spawn_x
        player_spawn_y = DungeonData.player_spawn_y
        stairs_spawn_x = DungeonData.stairs_spawn_x
        stairs_spawn_y = DungeonData.stairs_spawn_y

        map_array = self.tiles_to_numpy(DungeonData.list_tiles)

        corridor_length = 0.0
        for x in range(32):
            for y in range(32):
                tile = DungeonData.list_tiles[x][y]
                # Use same check as dungeon algorithm: & 0x3 == 1 means walkable floor
                # Corridors are floor tiles with room_index == 0xFF or 0xFE
                if (tile[0x0] & 0x3) == 1 and (tile[0x7] == 0xFF or tile[0x7] == 0xFE):
                    corridor_length += 1.0

        # Set spawn points as 5x5 areas
        if player_spawn_x != -1 and player_spawn_y != -1:
            # Create a 5x5 area around the spawn point
            for dy in range(-2, 3):  # -2 to +2 (5 pixels)
                for dx in range(-2, 3):  # -2 to +2 (5 pixels)
                    ny = player_spawn_y + dy
                    nx = player_spawn_x + dx
                    # Only set if within bounds and on a floor tile
                    if 0 <= ny < map_array.shape[0] and 0 <= nx < map_array.shape[1]:
                        if map_array[ny, nx] == 1:  # Only overwrite floor tiles
                            map_array[ny, nx] = 2
        
        if stairs_spawn_x != -1 and stairs_spawn_y != -1:
            # Create a 5x5 area around the stairs spawn point
            for dy in range(-2, 3):  # -2 to +2 (5 pixels)
                for dx in range(-2, 3):  # -2 to +2 (5 pixels)
                    ny = stairs_spawn_y + dy
                    nx = stairs_spawn_x + dx
                    # Only set if within bounds and on a floor tile
                    if 0 <= ny < map_array.shape[0] and 0 <= nx < map_array.shape[1]:
                        if map_array[ny, nx] == 1:  # Only overwrite floor tiles
                            map_array[ny, nx] = 3

        # Generate enemies based on difficulty
        # Pass original tiles for validation to ensure enemies only spawn on actual floor tiles
        enemies = self.generate_enemies(
            map_array,
            (player_spawn_x, player_spawn_y),
            (stairs_spawn_x, stairs_spawn_y),
            difficulty=difficulty,
            original_tiles=DungeonData.list_tiles
        )

        metadata = {
            'width': 32,
            'height': 32,
            'room_count': Properties.nb_rooms,
            'layout_type': layout_type,
            'complexity': complexity,
            'difficulty': difficulty,
            'corridor_length': corridor_length,  # Use calculated value
            'generation_params': json.dumps({
                'layout_type': layout_type,
                'room_count': room_count,
                'complexity': complexity,
                'difficulty': difficulty
            }),
            'player_spawn_x': int(player_spawn_x) if player_spawn_x != -1 else None,
            'player_spawn_y': int(player_spawn_y) if player_spawn_y != -1 else None,
            'stairs_spawn_x': int(stairs_spawn_x) if stairs_spawn_x != -1 else None,
            'stairs_spawn_y': int(stairs_spawn_y) if stairs_spawn_y != -1 else None,
            'enemies': json.dumps(enemies)
        }

        return map_array, metadata
    
    def tiles_to_numpy(self, tiles):
        height, width = len(tiles[0]), len(tiles)
        map_array = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                tile = tiles[x][y]
                # Use same check as dungeon algorithm: & 0x3 == 1 means walkable
                map_array[y, x] = 1 if (tile[0x0] & 0x3) == 1 else 0

        return map_array
    
    def map_array_to_image(self,
                           map_array: np.ndarray) -> Image.Image:
        
        height, width = map_array.shape
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        img_array[map_array == 0] = [0, 0, 0] #Black walls
        img_array[map_array == 1] = [139, 69, 19] #Brown floors
        img_array[map_array == 2] = [0, 255, 0] #Green player spawn
        img_array[map_array == 3] = [255, 0, 0] #Red stairs spawn

        img = Image.fromarray(img_array, mode='RGB')
        img = img.resize(self.image_size, Image.LANCZOS)

        return img

    def calculate_hash(self, 
                       map_array: np.ndarray) -> str:

        return hashlib.md5(map_array.tobytes()).hexdigest()
    
    def generate_dataset(self,
                         num_maps: int = 1000,
                         width_range: Tuple[int,int] = (20, 50),
                         height_range: Tuple[int, int] = (20, 50),
                         room_range: Tuple[int, int] = (3, 10),
                         complexity_range: Tuple[float, float] = (0.2, 0.8),
                         difficulty_distribution: Optional[Dict[str, float]] = None) -> Dataset:
        """
        Generate dataset with difficulty distribution.
        
        Args:
            difficulty_distribution: Dict like {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
                                    If None, uses equal distribution
        """
        # logger.info(f"Generating {num_maps} dungeon maps...")

        dataset_data = []
        seen_hashes = set()
        
        # Default difficulty distribution
        if difficulty_distribution is None:
            difficulty_distribution = {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}
        
        # Normalize distribution
        total = sum(difficulty_distribution.values())
        difficulty_distribution = {k: v/total for k, v in difficulty_distribution.items()}
        
        # Create difficulty list for random selection
        difficulty_list = []
        for difficulty, weight in difficulty_distribution.items():
            difficulty_list.extend([difficulty] * int(weight * num_maps))
        
        # Fill remaining slots
        import random
        while len(difficulty_list) < num_maps:
            difficulty_list.append('medium')
        
        random.shuffle(difficulty_list)

        # Keep generating until we have num_maps valid maps
        attempts = 0
        max_attempts = num_maps * 20  # Safety limit: allow up to 20x attempts
        
        while len(dataset_data) < num_maps and attempts < max_attempts:
            attempts += 1
            
            # Cycle through difficulty list to maintain distribution
            difficulty_idx = len(dataset_data) % len(difficulty_list)
            difficulty = difficulty_list[difficulty_idx]
            
            room_count = np.random.randint(*room_range)
            complexity = np.random.uniform(*complexity_range)

            try:
                layout_type = np.random.randint(1, 8)
                map_array, metadata = self.generate_map(
                    layout_type, 
                    room_count, 
                    complexity,
                    difficulty=difficulty
                )
                
                # Check playability: player to stairs AND player to all enemies
                # Parse enemies from metadata
                enemies_list = []
                if 'enemies' in metadata:
                    try:
                        enemies_list = json.loads(metadata['enemies'])
                    except:
                        enemies_list = []
                
                player_spawn = (
                    metadata.get('player_spawn_x', -1),
                    metadata.get('player_spawn_y', -1)
                )
                stairs_spawn = (
                    metadata.get('stairs_spawn_x', -1),
                    metadata.get('stairs_spawn_y', -1)
                )
                
                if not is_map_playable(map_array, enemies_list, player_spawn, stairs_spawn):
                    continue  # Skip unplayable maps and try again

                map_hash = self.calculate_hash(map_array)
                if map_hash in seen_hashes:
                    continue  # Skip duplicates and try again
                
                seen_hashes.add(map_hash)

                img = self.map_array_to_image(map_array)

                dataset_entry = {
                    'image': img,
                    'map_id': f"map_{len(dataset_data):06d}",
                    'width': metadata['width'],
                    'height': metadata['height'],
                    'complexity': metadata['complexity'],
                    'room_count': metadata['room_count'],
                    'corridor_length': metadata['corridor_length'],
                    'difficulty': metadata['difficulty'],
                    'generation_params': metadata['generation_params'],
                    'hash': map_hash,
                    'map_array': map_array,
                    'player_spawn_x': metadata.get('player_spawn_x', -1) or -1,
                    'player_spawn_y': metadata.get('player_spawn_y', -1) or -1,
                    'stairs_spawn_x': metadata.get('stairs_spawn_x', -1) or -1,
                    'stairs_spawn_y': metadata.get('stairs_spawn_y', -1) or -1,
                    'enemies': metadata.get('enemies', '[]')
                }

                dataset_data.append(dataset_entry)
                
                # Progress indicator
                if len(dataset_data) % 100 == 0:
                    print(f"Generated {len(dataset_data)}/{num_maps} valid maps (attempts: {attempts}, success rate: {len(dataset_data)/attempts*100:.1f}%)")
            
            except Exception as e:
                # logger.error(f"Error generating map: {e}")
                pass
        
        if len(dataset_data) < num_maps:
            print(f"Warning: Only generated {len(dataset_data)}/{num_maps} valid maps after {attempts} attempts")
            print(f"  Success rate: {len(dataset_data)/attempts*100:.1f}%")
            print(f"  This might indicate too many unplayable maps or duplicates")
            print(f"  Consider adjusting room_range or complexity_range")
        
        # logger.info(f"Successfully generated {len(dataset_data)} unique maps")

        dataset = Dataset.from_list(dataset_data, features=self.features)
        return dataset
    
    def save_dataset(self, 
                     dataset: Dataset, 
                     dataset_name: str = "mysterydungeonmaps"):
        
        save_path = self.output_dir / dataset_name
        dataset.save_to_disk(str(save_path))
        # logger.info(f"Dataset saved to {save_path}")
        return save_path
    
    def upload_to_hub(self,
                      dataset: Dataset,
                      repo_id: str):
        try:
            dataset.push_to_hub(repo_id)
            # logger.info(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            # logger.error(f"Error uploading to Hub: {e}")
            raise
    
    def create_train_val_split(self,
                               dataset: Dataset,
                               val_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
        return split_dataset['train'], split_dataset['test']

def main():
    generator = MysteryDungeonMapGenerator(
        output_dir="./mystery_dungeon_data",
        image_size=(256, 256),
        map_size=(32, 32),
        hf_repo_id="teamgas/mysterydungeondata"
    )

    dataset = generator.generate_dataset(
        num_maps = 5000,
        width_range = (20, 50),
        height_range = (20, 50),
        room_range = (3, 6),
        complexity_range = (0.2, 0.8),
        difficulty_distribution = {'medium': 1.0}
    )

    train_dataset, val_dataset = generator.create_train_val_split(dataset, val_split=0.2)

    train_path = generator.save_dataset(train_dataset, "mystery_dungeon_train")
    val_path = generator.save_dataset(val_dataset, "mystery_dungeon_val")

    generator.upload_to_hub(dataset, "teamgas/mysterydungeondata")

    print(f"Dataset generation complete")
    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

if __name__ == "__main__":
    main()



