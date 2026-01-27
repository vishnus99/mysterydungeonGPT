"""
Map Viewer for Mystery Dungeon Maps
Loads maps from HuggingFace datasets and displays them in various formats
"""

import numpy as np
import pandas as pd
import json
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import argparse
import os

from src.playablemap import playablebfs

class MysteryDungeonMapViewer:
    def __init__(self):
        # ASCII characters for different tile types
        self.ascii_chars = {
            0: '#',  # Wall
            1: '.',  # Floor
            2: 'P',  # Player spawn
            3: 'S',  # Stairs spawn
        }
        
        # Colors for matplotlib visualization
        self.colors = {
            0: '#000000',  # Black walls
            1: '#8B4513',  # Brown floors
            2: '#00FF00',  # Green player spawn
            3: '#FF0000',  # Red stairs spawn
        }
        
        # Enemy colors by type
        self.enemy_colors = {
            'weak': '#FFA07A',      # Light salmon
            'basic': '#FF6B6B',      # Red
            'aggressive': '#DC143C', # Crimson
            'patrol': '#8B0000',     # Dark red
            'elite': '#4B0082',      # Indigo
            'boss': '#FF00FF'        # Magenta
        }
    
    def load_dataset_from_huggingface(self, dataset_name: str) -> pd.DataFrame:
        """Load dataset from HuggingFace Hub"""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        # Convert to pandas for easier manipulation
        df = pd.DataFrame(dataset['train'])
        print(f"Loaded {len(df)} maps")
        return df
    
    def load_dataset_from_local(self, parquet_path: str) -> pd.DataFrame:
        """Load dataset from local parquet file"""
        print(f"Loading local dataset: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} maps")
        return df
    
    def load_map_from_json(self, json_path: str) -> Tuple[np.ndarray, dict]:
        """
        Load a map from a JSON file (coordinate-based or grid-based format).
        
        Args:
            json_path: Path to JSON file
        
        Returns:
            Tuple of (map_array, metadata_dict)
        """
        with open(json_path, 'r') as f:
            map_data = json.load(f)
        
        # Check if it's coordinate-based format (walkable_tiles)
        if 'walkable_tiles' in map_data:
            from mysterydungeonGPT.helpers import coordinates_to_grid
            
            width = map_data.get('width', 56)
            height = map_data.get('height', 32)
            walkable_coords = map_data['walkable_tiles']
            
            # Convert coordinates to grid
            grid = coordinates_to_grid(walkable_coords, width=width, height=height)
            
            # Initialize spawn points
            px, py = -1, -1
            sx, sy = -1, -1
            
            # Add spawn points (convert to grid coordinates)
            if 'player_spawn' in map_data:
                spawn = map_data['player_spawn']
                if isinstance(spawn, list):
                    px, py = spawn[0], spawn[1]
                else:
                    px, py = spawn.get('x', -1), spawn.get('y', -1)
                if 0 <= px < width and 0 <= py < height:
                    grid[py, px] = 2  # Player spawn
            
            if 'stairs_spawn' in map_data:
                spawn = map_data['stairs_spawn']
                if isinstance(spawn, list):
                    sx, sy = spawn[0], spawn[1]
                else:
                    sx, sy = spawn.get('x', -1), spawn.get('y', -1)
                if 0 <= sx < width and 0 <= sy < height:
                    grid[sy, sx] = 3  # Stairs spawn
            
            # Create metadata dict compatible with viewer
            metadata = {
                'width': width,
                'height': height,
                'player_spawn_x': px,
                'player_spawn_y': py,
                'stairs_spawn_x': sx,
                'stairs_spawn_y': sy,
                'enemies': map_data.get('enemies', []),
                'difficulty': map_data.get('difficulty', 'unknown'),
                'map_id': os.path.basename(json_path).replace('.json', '')
            }
            
            return grid, metadata
        
        # Check if it's grid-based format (tiles array)
        elif 'tiles' in map_data:
            map_array = np.array(map_data['tiles'], dtype=np.uint8)
            
            # Extract metadata
            metadata = {
                'width': map_data.get('width', map_array.shape[1]),
                'height': map_data.get('height', map_array.shape[0]),
                'player_spawn_x': -1,
                'player_spawn_y': -1,
                'stairs_spawn_x': -1,
                'stairs_spawn_y': -1,
                'enemies': map_data.get('enemies', []),
                'difficulty': map_data.get('difficulty', 'unknown'),
                'map_id': os.path.basename(json_path).replace('.json', '')
            }
            
            # Find spawn points in the grid
            player_spawns = np.where(map_array == 2)
            stairs_spawns = np.where(map_array == 3)
            
            if len(player_spawns[0]) > 0:
                metadata['player_spawn_y'] = player_spawns[0][0]
                metadata['player_spawn_x'] = player_spawns[1][0]
            
            if len(stairs_spawns[0]) > 0:
                metadata['stairs_spawn_y'] = stairs_spawns[0][0]
                metadata['stairs_spawn_x'] = stairs_spawns[1][0]
            
            return map_array, metadata
        
        else:
            raise ValueError(f"Unknown map format in {json_path}. Expected 'walkable_tiles' or 'tiles' field.")
    
    def load_maps_from_json_directory(self, json_dir: str) -> pd.DataFrame:
        """
        Load all JSON map files from a directory.
        
        Args:
            json_dir: Directory containing JSON map files
        
        Returns:
            DataFrame compatible with the viewer
        """
        json_dir_path = Path(json_dir)
        json_files = list(json_dir_path.glob('*.json'))
        
        # Filter out map_index.json
        json_files = [f for f in json_files if f.name != 'map_index.json']
        
        if not json_files:
            raise ValueError(f"No JSON map files found in {json_dir}")
        
        print(f"Loading {len(json_files)} maps from {json_dir}")
        
        maps_data = []
        for json_file in json_files:
            try:
                map_array, metadata = self.load_map_from_json(str(json_file))
                
                # Convert map_array to PIL Image for compatibility with viewer
                # Create a simple RGB representation
                height, width = map_array.shape
                img_array = np.zeros((height, width, 3), dtype=np.uint8)
                
                for y in range(height):
                    for x in range(width):
                        tile = map_array[y, x]
                        if tile == 0:
                            img_array[y, x] = [0, 0, 0]  # Black wall
                        elif tile == 1:
                            img_array[y, x] = [139, 69, 19]  # Brown floor
                        elif tile == 2:
                            img_array[y, x] = [0, 255, 0]  # Green player
                        elif tile == 3:
                            img_array[y, x] = [255, 0, 0]  # Red stairs
                
                img = Image.fromarray(img_array)
                
                # Create a row compatible with the viewer's DataFrame format
                row = {
                    'image': img,
                    'map_id': metadata.get('map_id', json_file.stem),
                    'width': metadata['width'],
                    'height': metadata['height'],
                    'player_spawn_x': metadata.get('player_spawn_x', -1),
                    'player_spawn_y': metadata.get('player_spawn_y', -1),
                    'stairs_spawn_x': metadata.get('stairs_spawn_x', -1),
                    'stairs_spawn_y': metadata.get('stairs_spawn_y', -1),
                    'enemies': json.dumps(metadata.get('enemies', [])),
                    'difficulty': metadata.get('difficulty', 'unknown'),
                    'room_count': 0,  # Not available in generated maps
                    'complexity': 0.0,  # Not available in generated maps
                    'corridor_length': 0  # Not available in generated maps
                }
                maps_data.append(row)
                
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        if not maps_data:
            raise ValueError(f"No valid maps loaded from {json_dir}")
        
        df = pd.DataFrame(maps_data)
        print(f"Successfully loaded {len(df)} maps")
        return df
    
    def extract_map_from_image(self, image_pil: Image.Image) -> np.ndarray:
        """Extract map array from PIL Image"""
        # Convert image back to numpy array
        img_array = np.array(image_pil)
        
        # If it's RGB (3 channels), check for spawn points and convert properly
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB image - check for spawn point colors
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
                        # Default to floor if it's not pure black
                        map_array[y, x] = 1 if (r + g + b) > 0 else 0
        elif len(img_array.shape) == 2:
            # Grayscale image
            normalized = img_array.astype(np.float32) / 255.0
            map_array = (normalized > 0.5).astype(np.uint8)
        else:
            # If RGB, convert to grayscale first
            gray = np.mean(img_array, axis=2)
            normalized = gray.astype(np.float32) / 255.0
            map_array = (normalized > 0.5).astype(np.uint8)
        
        return map_array
    
    def display_ascii_map(self, map_array: np.ndarray, title: str = "Map", 
                          save_to_file: str = None, path: Optional[List[Tuple[int, int]]] = None,
                          enemies: Optional[List[Dict]] = None):
        """Display map as ASCII art"""
        
        ascii_output = []
        ascii_output.append(f"\n{title}")
        ascii_output.append("=" * (map_array.shape[1] + 2))
        
        path_set = set(path) if path else set()
        
        # Create enemy position set for quick lookup
        enemy_positions = set()
        if enemies:
            # Handle if enemies is a JSON string
            if isinstance(enemies, str):
                try:
                    enemies = json.loads(enemies)
                except:
                    enemies = []
            if isinstance(enemies, list):
                for enemy in enemies:
                    if isinstance(enemy, dict) and 'x' in enemy and 'y' in enemy:
                        enemy_positions.add((enemy['x'], enemy['y']))

        for y, row in enumerate(map_array):
            ascii_row_chars = []
            for x, tile in enumerate(row):
                if (x, y) in enemy_positions:
                    # Show enemy as 'E'
                    ascii_row_chars.append('E')
                elif (x, y) in path_set and tile not in (2, 3):
                    ascii_row_chars.append('@')          # blue path marker in text view
                else:
                    ascii_row_chars.append(self.ascii_chars.get(tile, '?'))
            ascii_output.append(f"|{''.join(ascii_row_chars)}|")
        
        ascii_output.append("=" * (map_array.shape[1] + 2))
        ascii_output.append(f"Size: {map_array.shape[1]}x{map_array.shape[0]}")
        ascii_output.append(f"Floor tiles: {np.sum(map_array == 1)}")
        ascii_output.append(f"Wall tiles: {np.sum(map_array == 0)}")
        if enemies and len(enemy_positions) > 0:
            ascii_output.append(f"Enemies: {len(enemy_positions)}")
        
        for line in ascii_output:
            print(line)
        
        if save_to_file:
            with open(save_to_file, 'w') as f:
                f.write('\n'.join(ascii_output))
            print(f"ASCII map saved to: {save_to_file}")
    
    def display_matplotlib_map(self, map_array: np.ndarray, title: str = "Map", 
                               figsize: Tuple[int, int] = (10, 8), save_to_file: str = None,
                               path: Optional[List[Tuple[int, int]]] = None,
                               enemies: Optional[List[Dict]] = None):
        """Display map using matplotlib"""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colored_map = np.zeros((*map_array.shape, 3))
        for tile_type, color in self.colors.items():
            mask = map_array == tile_type
            colored_map[mask] = plt.matplotlib.colors.to_rgb(color)
        
        if path:
            path_color = plt.matplotlib.colors.to_rgb('#1E90FF')  # blue
            for x, y in path:
                if 0 <= y < map_array.shape[0] and 0 <= x < map_array.shape[1]:
                    if map_array[y, x] not in (2, 3):            # keep spawn/stairs visible
                        colored_map[y, x] = path_color

        # Draw enemies
        if enemies:
            # Handle if enemies is a JSON string
            if isinstance(enemies, str):
                try:
                    enemies = json.loads(enemies)
                except:
                    enemies = []
            if isinstance(enemies, list):
                for enemy in enemies:
                    if isinstance(enemy, dict) and 'x' in enemy and 'y' in enemy:
                        x, y = enemy['x'], enemy['y']
                        if 0 <= y < map_array.shape[0] and 0 <= x < map_array.shape[1]:
                            # Simple red color for all enemies
                            enemy_color = '#FF6B6B'
                            colored_map[y, x] = plt.matplotlib.colors.to_rgb(enemy_color)

        ax.imshow(colored_map)
        ax.set_title(title)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        ax.set_xticks(range(0, map_array.shape[1], 5))
        ax.set_yticks(range(0, map_array.shape[0], 5))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_to_file:
            plt.savefig(save_to_file, dpi=300, bbox_inches='tight')
            print(f"Map image saved to: {save_to_file}")
        else:
            plt.show()
        
        plt.close()
    
    def save_map_as_numpy(self, map_array: np.ndarray, filename: str):
        """Save map as numpy array file"""
        np.save(filename, map_array)
        print(f"Map array saved to: {filename}")
    
    def save_map_as_csv(self, map_array: np.ndarray, filename: str):
        """Save map as CSV file"""
        np.savetxt(filename, map_array, delimiter=',', fmt='%d')
        print(f"Map CSV saved to: {filename}")
    
    def save_map_as_png(self, map_array: np.ndarray, filename: str, 
                       tile_size: int = 10):
        """Save map as PNG image with custom tile size"""
        # Scale up the map for better visibility
        scaled_map = np.repeat(np.repeat(map_array, tile_size, axis=0), tile_size, axis=1)
        
        # Convert to PIL Image
        img = Image.fromarray(scaled_map * 255, mode='L')
        img.save(filename)
        print(f"Map PNG saved to: {filename}")
    
    def export_map_for_rom_hacking(self, map_array: np.ndarray, filename: str):
        """Export map in format suitable for ROM hacking"""
        # Convert to Pokemon Red Rescue Team format
        pmd_tiles = np.zeros_like(map_array, dtype=np.uint8)
        pmd_tiles[map_array == 0] = 0x00  # Wall
        pmd_tiles[map_array == 1] = 0x01  # Floor
        
        # Save as binary file
        with open(filename, 'wb') as f:
            f.write(pmd_tiles.tobytes())
        
        print(f"ROM hacking format saved to: {filename}")
    
    def display_map_stats(self, map_array: np.ndarray, metadata: dict = None):
        """Display statistics about the map"""
        print("\nMap Statistics:")
        print("-" * 30)
        print(f"Dimensions: {map_array.shape[1]} x {map_array.shape[0]}")
        print(f"Total tiles: {map_array.size}")
        print(f"Floor tiles: {np.sum(map_array == 1)} ({np.sum(map_array == 1)/map_array.size*100:.1f}%)")
        print(f"Wall tiles: {np.sum(map_array == 0)} ({np.sum(map_array == 0)/map_array.size*100:.1f}%)")
        
        if metadata:
            print(f"\nMetadata:")
            for key, value in metadata.items():
                if key != 'image':  # Skip the image data
                    print(f"  {key}: {value}")
    
    def view_single_map(self, df: pd.DataFrame, index: int = 0, 
                       display_format: str = "both", output_dir: str = None):
        """View a single map from the dataset"""
        if index >= len(df):
            print(f"Index {index} out of range. Dataset has {len(df)} maps.")
            return
        
        row = df.iloc[index]
        
        # Use stored map_array if available, otherwise extract from image
        if 'map_array' in row and row['map_array'] is not None:
            # Handle if it's stored as a string or numpy array
            if isinstance(row['map_array'], (str, bytes)):
                # If it's stored as a string representation, you may need to parse it
                # For now, fall back to image extraction
                map_array = self.extract_map_from_image(row['image'])
            else:
                map_array = np.array(row['map_array'])
        else:
            # Fall back to extracting from image
            map_array = self.extract_map_from_image(row['image'])
        
        path = playablebfs(map_array)
        
        # Extract enemies from metadata if available
        enemies = None
        if 'enemies' in row and row['enemies'] is not None:
            enemies = row['enemies']
        
        # Display map
        title = f"Map {index} (ID: {row.get('map_id', 'Unknown')})"
        if 'difficulty' in row:
            title += f" - Difficulty: {row['difficulty']}"
        
        # Create output directory if specified
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            base_filename = f"{output_dir}/map_{index:06d}"
        else:
            base_filename = None
        
        if display_format in ["ascii", "both"]:
            ascii_file = f"{base_filename}.txt" if base_filename else None
            self.display_ascii_map(map_array, title, ascii_file, path=path, enemies=enemies)
        
        if display_format in ["matplotlib", "both"]:
            img_file = f"{base_filename}.png" if base_filename else None
            self.display_matplotlib_map(map_array, title, save_to_file=img_file, path=path, enemies=enemies)
        
        # Save additional formats if output directory specified
        if output_dir:
            self.save_map_as_numpy(map_array, f"{base_filename}.npy")
            self.save_map_as_csv(map_array, f"{base_filename}.csv")
            self.save_map_as_png(map_array, f"{base_filename}_scaled.png")
            self.export_map_for_rom_hacking(map_array, f"{base_filename}.bin")
        
        # Show stats
        metadata = {k: v for k, v in row.items() if k != 'image'}
        self.display_map_stats(map_array, metadata)
    
    def view_multiple_maps(self, df: pd.DataFrame, indices: List[int], 
                          display_format: str = "matplotlib"):
        """View multiple maps in a grid"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(indices[:6]):  # Show max 6 maps
            if idx >= len(df):
                break
                
            row = df.iloc[idx]
            
            # Use stored map_array if available
            if 'map_array' in row and row['map_array'] is not None:
                if isinstance(row['map_array'], (str, bytes)):
                    map_array = self.extract_map_from_image(row['image'])
                else:
                    map_array = np.array(row['map_array'])
            else:
                map_array = self.extract_map_from_image(row['image'])
            
            path = playablebfs(map_array)
            
            # Create colored map
            colored_map = np.zeros((*map_array.shape, 3))
            for tile_type, color in self.colors.items():
                mask = map_array == tile_type
                colored_map[mask] = plt.matplotlib.colors.to_rgb(color)

            if path:
                path_color = plt.matplotlib.colors.to_rgb('#1E90FF')
                for x, y in path:
                    if 0 <= y < map_array.shape[0] and 0 <= x < map_array.shape[1]:
                        if map_array[y, x] not in (2, 3):
                            colored_map[y, x] = path_color
            
            axes[i].imshow(colored_map)
            axes[i].set_title(f"Map {idx}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
        
        # Hide unused subplots
        for i in range(len(indices), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def search_maps(self, df: pd.DataFrame, **criteria):
        """Search for maps matching certain criteria"""
        mask = pd.Series([True] * len(df))
        
        for key, value in criteria.items():
            if key in df.columns:
                if isinstance(value, tuple):  # Range search
                    mask &= (df[key] >= value[0]) & (df[key] <= value[1])
                else:  # Exact match
                    mask &= (df[key] == value)
        
        matching_indices = df[mask].index.tolist()
        print(f"Found {len(matching_indices)} maps matching criteria: {criteria}")
        return matching_indices
    
    def interactive_viewer(self, df: pd.DataFrame):
        """Interactive map viewer"""
        print("Interactive Map Viewer")
        print("Commands:")
        print("  view <index> - View map at index")
        print("  search <criteria> - Search for maps")
        print("  random - View random map")
        print("  stats - Show dataset statistics")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower().split()
                
                if not command:
                    continue
                
                if command[0] == "quit":
                    break
                elif command[0] == "view" and len(command) > 1:
                    index = int(command[1])
                    self.view_single_map(df, index)
                elif command[0] == "random":
                    import random
                    index = random.randint(0, len(df) - 1)
                    self.view_single_map(df, index)
                elif command[0] == "stats":
                    self.show_dataset_stats(df)
                elif command[0] == "search":
                    # Simple search by room count
                    if len(command) > 1:
                        room_count = int(command[1])
                        indices = self.search_maps(df, room_count=room_count)
                        if indices:
                            self.view_multiple_maps(df, indices[:6])
                else:
                    print("Unknown command")
                    
            except (ValueError, IndexError) as e:
                print(f"Error: {e}")
    
    def show_dataset_stats(self, df: pd.DataFrame):
        """Show overall dataset statistics"""
        print("\nDataset Statistics:")
        print("=" * 40)
        print(f"Total maps: {len(df)}")
        
        if 'room_count' in df.columns:
            print(f"Room count range: {df['room_count'].min()} - {df['room_count'].max()}")
            print(f"Average rooms: {df['room_count'].mean():.1f}")
        
        if 'complexity' in df.columns:
            print(f"Complexity range: {df['complexity'].min():.2f} - {df['complexity'].max():.2f}")
            print(f"Average complexity: {df['complexity'].mean():.2f}")
        
        if 'width' in df.columns and 'height' in df.columns:
            print(f"Map dimensions: {df['width'].iloc[0]} x {df['height'].iloc[0]}")
    
    def export_all_maps(self, df: pd.DataFrame, output_dir: str):
        """Export all maps in the dataset to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Exporting {len(df)} maps to {output_dir}...")
        
        for i in range(len(df)):
            if i % 10 == 0:
                print(f"Processing map {i}/{len(df)}")
            
            row = df.iloc[i]
            map_array = self.extract_map_from_image(row['image'])
            
            base_filename = f"{output_dir}/map_{i:06d}"
            
            # Save in all formats
            self.save_map_as_numpy(map_array, f"{base_filename}.npy")
            self.save_map_as_csv(map_array, f"{base_filename}.csv")
            self.save_map_as_png(map_array, f"{base_filename}_scaled.png")
            self.export_map_for_rom_hacking(map_array, f"{base_filename}.bin")
            
            # Save metadata
            metadata = {k: v for k, v in row.items() if k != 'image'}
            with open(f"{base_filename}_metadata.txt", 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
        
        print(f"Export complete! All maps saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="View Mystery Dungeon Maps")
    parser.add_argument("--dataset", help="HuggingFace dataset name")
    parser.add_argument("--local", help="Local parquet file path")
    parser.add_argument("--json-file", help="Single JSON map file to view")
    parser.add_argument("--json-dir", help="Directory containing JSON map files")
    parser.add_argument("--index", type=int, default=0, help="Map index to view")
    parser.add_argument("--format", choices=["ascii", "matplotlib", "both"], 
                       default="both", help="Display format")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--output-dir", help="Directory to save output files")
    parser.add_argument("--export-all", action="store_true", 
                       help="Export all maps in dataset to files")
    
    args = parser.parse_args()
    
    viewer = MysteryDungeonMapViewer()
    
    # Load dataset
    if args.dataset:
        df = viewer.load_dataset_from_huggingface(args.dataset)
    elif args.local:
        df = viewer.load_dataset_from_local(args.local)
    elif args.json_file:
        # Load single JSON file - create a temporary DataFrame
        map_array, metadata = viewer.load_map_from_json(args.json_file)
        
        # Convert to image for DataFrame compatibility
        height, width = map_array.shape
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                tile = map_array[y, x]
                if tile == 0:
                    img_array[y, x] = [0, 0, 0]
                elif tile == 1:
                    img_array[y, x] = [139, 69, 19]
                elif tile == 2:
                    img_array[y, x] = [0, 255, 0]
                elif tile == 3:
                    img_array[y, x] = [255, 0, 0]
        
        img = Image.fromarray(img_array)
        df = pd.DataFrame([{
            'image': img,
            'map_id': metadata.get('map_id', Path(args.json_file).stem),
            'width': metadata['width'],
            'height': metadata['height'],
            'player_spawn_x': metadata.get('player_spawn_x', -1),
            'player_spawn_y': metadata.get('player_spawn_y', -1),
            'stairs_spawn_x': metadata.get('stairs_spawn_x', -1),
            'stairs_spawn_y': metadata.get('stairs_spawn_y', -1),
            'enemies': json.dumps(metadata.get('enemies', [])),
            'difficulty': metadata.get('difficulty', 'unknown'),
            'room_count': 0,
            'complexity': 0.0,
            'corridor_length': 0
        }])
    elif args.json_dir:
        df = viewer.load_maps_from_json_directory(args.json_dir)
    else:
        print("Please specify one of: --dataset, --local, --json-file, or --json-dir")
        return
    
    if args.interactive:
        viewer.interactive_viewer(df)
    elif args.export_all:
        if not args.output_dir:
            print("--output-dir required for --export-all")
            return
        viewer.export_all_maps(df, args.output_dir)
    else:
        viewer.view_single_map(df, args.index, args.format, args.output_dir)

if __name__ == "__main__":
    main()