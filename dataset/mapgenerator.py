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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.warning("Invalid player spawn position")
        return False
    
    if stairs_spawn[0] == -1 or stairs_spawn[1] == -1:
        logger.warning("Invalid stairs spawn position")
        return False
    
    if not is_reachable_from_spawn(map_array, player_spawn, stairs_spawn):
        logger.debug("Player spawn to stairs is not reachable")
        return False
    
    # Check player spawn to each enemy spawn
    # If there are no enemies, the map is still playable (player can reach stairs)
    if not enemies:
        return True
    
    for enemy in enemies:
        if 'x' not in enemy or 'y' not in enemy:
            logger.warning(f"Enemy missing x or y coordinates: {enemy}")
            continue
        
        enemy_pos = (enemy['x'], enemy['y'])
        if not is_reachable_from_spawn(map_array, player_spawn, enemy_pos):
            logger.debug(f"Player spawn to enemy at ({enemy_pos[0]}, {enemy_pos[1]}) is not reachable")
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
        Generate enemies based on difficulty level.
        
        Args:
            map_array: The map array (0=wall, 1=floor, 2=player spawn, 3=stairs spawn)
            player_spawn: (x, y) player spawn position
            stairs_spawn: (x, y) stairs spawn position
            difficulty: 'easy', 'medium', or 'hard'
            original_tiles: Original DungeonData.list_tiles for validation (optional)
        
        Returns:
            List of enemy dictionaries with 'x', 'y', 'type', 'hp', 'max_hp', and 'damage' fields
        """
        import random
        
        # Get difficulty config
        config = self.DIFFICULTY_CONFIGS.get(difficulty, self.DIFFICULTY_CONFIGS['medium'])
        height, width = map_array.shape
        
        # Get all valid floor tile positions
        # Exclude spawn areas (5x5 areas around spawn points)
        player_spawn_x, player_spawn_y = player_spawn
        stairs_spawn_x, stairs_spawn_y = stairs_spawn
        
        floor_positions = []
        for y in range(height):
            for x in range(width):
                # BOTH checks must pass: original_tiles AND map_array must agree it's a floor
                # This ensures we never place enemies on walls
                
                # Check 1: map_array must be floor (value 1), not wall (0) or spawn area (2, 3)
                if map_array[y, x] != 1:
                    continue  # Skip walls and spawn areas
                
                # Check 2: original_tiles must also say it's a floor
                if original_tiles is not None:
                    try:
                        # original_tiles is indexed as [x][y], dimensions are 56x32
                        if x >= len(original_tiles) or y >= len(original_tiles[0]):
                            # Out of bounds in original_tiles, skip
                            continue
                        
                        tile = original_tiles[x][y]
                        # Tile is walkable if lower 2 bits are 0x1 (matches dungeon algorithm check)
                        if (tile[0x0] & 0x3) != 1:
                            # Not a floor in original_tiles, skip
                            continue
                    except (IndexError, AttributeError, TypeError, KeyError) as e:
                        # If validation fails, skip this position to be safe
                        logger.debug(f"Error validating tile at ({x}, {y}): {e}")
                        continue
                
                # Third check: Exclude spawn areas (5x5 areas around spawn points)
                in_player_area = False
                if player_spawn_x != -1 and player_spawn_y != -1:
                    if (abs(x - player_spawn_x) <= 2 and abs(y - player_spawn_y) <= 2):
                        in_player_area = True
                
                in_stairs_area = False
                if stairs_spawn_x != -1 and stairs_spawn_y != -1:
                    if (abs(x - stairs_spawn_x) <= 2 and abs(y - stairs_spawn_y) <= 2):
                        in_stairs_area = True
                
                # Only add if it's a floor tile and not in any spawn area
                if not in_player_area and not in_stairs_area:
                    floor_positions.append((x, y))
        
        # Filter to only reachable positions from player spawn
        if player_spawn_x != -1 and player_spawn_y != -1:
            floor_positions = [pos for pos in floor_positions 
                             if is_reachable_from_spawn(map_array, player_spawn, pos)]
        
        if len(floor_positions) == 0:
            logger.warning(f"No valid reachable floor positions found for enemy placement. Map size: {width}x{height}, Player spawn: {player_spawn}, Stairs spawn: {stairs_spawn}")
            return []
        
        # Calculate enemy count based on difficulty
        floor_tiles = len(floor_positions)
        logger.debug(f"Found {floor_tiles} valid floor positions for enemy placement")
        
        # Method 1: Use density-based calculation
        density_count = max(1, int(floor_tiles * config['enemy_density']))
        
        # Method 2: Use range-based calculation
        min_count, max_count = config['enemy_count_range']
        range_count = random.randint(min_count, max_count)
        
        # Use the smaller of the two to ensure we don't exceed available space
        num_enemies = min(density_count, range_count, len(floor_positions))
        
        # Safety check: ensure we have positions to select from
        if num_enemies <= 0 or not floor_positions:
            logger.warning(f"Cannot generate enemies: num_enemies={num_enemies}, available_positions={len(floor_positions)}")
            return []
        
        # Randomly select positions (already validated as reachable floor tiles)
        selected_positions = random.sample(floor_positions, num_enemies)
        
        # Enemy type weights (can be customized per difficulty)
        enemy_type_weights = {
            'easy': {'basic': 0.7, 'weak': 0.3},
            'medium': {'basic': 0.4, 'aggressive': 0.4, 'patrol': 0.2},
            'hard': {'aggressive': 0.3, 'patrol': 0.3, 'elite': 0.3, 'boss': 0.1}
        }
        
        weights = enemy_type_weights.get(difficulty, enemy_type_weights['medium'])
        enemy_types = config['enemy_types']
        
        # Ensure all enemy types have weights (fallback to equal distribution if missing)
        weight_list = []
        for enemy_type in enemy_types:
            weight = weights.get(enemy_type, 0.1)
            weight_list.append(max(weight, 0.01))  # Ensure minimum weight to avoid zero
        
        # Enemy stats by type
        enemy_stats = {
            'weak': {'hp': 5, 'damage': 1},
            'basic': {'hp': 10, 'damage': 2},
            'aggressive': {'hp': 15, 'damage': 3},
            'patrol': {'hp': 12, 'damage': 2},
            'elite': {'hp': 25, 'damage': 5},
            'boss': {'hp': 50, 'damage': 8}
        }
        
        # Create enemies from validated positions (already checked for reachability and floor tiles)
        enemies = []
        for x, y in selected_positions:
            # Weighted random selection of enemy type
            enemy_type = random.choices(enemy_types, weights=weight_list)[0]
            stats = enemy_stats.get(enemy_type, enemy_stats['basic'])
            
            enemies.append({
                'x': int(x),
                'y': int(y),
                'type': enemy_type,
                'hp': stats['hp'],
                'max_hp': stats['hp'],
                'damage': stats['damage']
            })
        
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
        for x in range(56):
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
            'width': 56,
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
        logger.info(f"Generating {num_maps} dungeon maps...")

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

        for i in range(num_maps):
            # width = np.random.randint(*width_range)
            # height = np.random.randint(*height_range)
            room_count = np.random.randint(*room_range)
            complexity = np.random.uniform(*complexity_range)
            difficulty = difficulty_list[i]  # Get difficulty for this map

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
                    print(f'Skipping unplayable map: Map {i} (unreachable enemy or stairs)')
                    continue

                map_hash = self.calculate_hash(map_array)
                if map_hash in seen_hashes:
                    continue
                seen_hashes.add(map_hash)

                img = self.map_array_to_image(map_array)

                dataset_entry = {
                    'image': img,
                    'map_id': f"map_{i:06d}",
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
            
            except Exception as e:
                logger.error(f"Error generating map {i}: {e}")
        
        logger.info(f"Successfully generated {len(dataset_data)} unique maps")

        dataset = Dataset.from_list(dataset_data, features=self.features)
        return dataset
    
    def save_dataset(self, 
                     dataset: Dataset, 
                     dataset_name: str = "mysterydungeonmaps"):
        
        save_path = self.output_dir / dataset_name
        dataset.save_to_disk(str(save_path))
        logger.info(f"Dataset saved to {save_path}")
        return save_path
    
    def upload_to_hub(self,
                      dataset: Dataset,
                      repo_id: str):
        try:
            dataset.push_to_hub(repo_id)
            logger.info(f"Dataset uploaded to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            logger.error(f"Error uploading to Hub: {e}")
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
        num_maps = 1000,
        width_range = (20, 50),
        height_range = (20, 50),
        room_range = (3, 10),
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



