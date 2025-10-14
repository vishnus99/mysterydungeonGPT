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
try:
    from dungeon_eos.RandomGen import *
    from dungeon_eos.DungeonAlgorithm import *
except ModuleNotFoundError:
    import sys
    sys.path.append("..")
    from dungeon_eos.RandomGen import *
    from dungeon_eos.DungeonAlgorithm import *

#Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MysteryDungeonMapGenerator: 
    def __init__(self,
                 output_dir: str = "./maps",
                 map_size: Tuple[int,int] = [256,256],
                 hf_repo_id: str = "vishnusm/mysterydungeonmaps"):

        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.hf_repo_id = hf_repo_id

        #Output dirs
        self.images_dir = self.output / "images"
        self.metadata_dir = self.output_dir / "metadata"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def generate_map(self,
                     width: int = 32,
                     height: int = 32,
                     room_count: int = 5,
                     complexity: float = 0.5) -> Tuple[np.ndarray, Dict]:
        try:
            dungeon = DungeonAlgorithm(width, height)
            rooms = dungeon.generate_rooms(room_count)
            corridors = dungeon.generate_corridors(rooms, complexity)
            map_array = dungeon.create_map(room, corridors)

            actual_room_count = len(rooms)
            corridor_length = sum(len(corridor) for corridor in corridors)

            metadata = {
                'width': width,
                'height': height,
                'room_count': actual_room_count,
                'corridor_length': corridor_length,
                'complexity': complexity,
                'generation_params': json.dumps({
                    'width': width,
                    'height': height,
                    'room_count': room_count,
                    'complexity': complexity
                })
            }

            return map_array, metadata

        except Exception as e:
            logger.error(f"Error generating map: {e}")
            raise
    
    def map_array_to_image(self,
                           map_array: np.ndarray) -> Image.Image:

        normalized = ((map_array - map_array.min()) / (map_array.max() - map_array.min()) * 255).astype(np.uint8)

        img = Image.fromarray(normalized, mode = 'L')

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
                         complexity_range: Tuple[float, float] = (0.2, 0.8)) -> Dataset:
        
        logger.info("Generating {num_maps} dungeon maps...")

        dataset_data = []
        seen_hashes = set()

        for i in range(num_maps):
            width = np.random.randint(*width_range)
            height = np.random.randint(*height_range)
            room_count = np.random.randint(*room_range)
            complexity = np.random.randint(*complexity_range)

            try:
                map_array, metadata = self.generate_map(width, height, room_count, complexity)

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
                    'generation_params': metadata['generation_params']
                }

                dataset_data.append(dataset_entry)
            
            except Exception as e:
                logger.error(f"Error generating map {i}: {e}")
        
        logger.info(f"Successfully generated {len(dataset_data)} unique maps")

        dataset = Dataset.from_list(dataset_data, features=self.features)
    
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
                               dataset, Dataset,
                               val_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        split_dataset = dataset.train_test_split(test_size=val_split, seed=42)
        return split_dataset['train'], split_dataset['test']

    def main():
        generator = MysteryDungeonMapGenerator(
            output_dir = "./mystery_dungeon_data",
            image_size = (256,256),
            hf_repo_id="vishnusm/mysterydungeonmaps"
        )

        dataset = generator.generate_dataset(
            num_maps = 1000,
            width_range = (20, 50),
            height_range = (20, 50),
            room_range = (3, 10),
            complexity_range = (0.2, 0.8)
        )

        train_dataset, val_dataset = generator.create_train_val_split(dataset, val_split=0.2)

        train_path = generator.save_dataset(train_dataset, "mystery_dungeon_train")
        val_path = generator.save_dataset(val_dataset, "mysterdy_dungeon_val")

        print(f"Dataset generation complete")
        print(f"Total samples: {len(dataset)}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

if __name__ == "__main__":
    main()








