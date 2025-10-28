"""
Map Viewer for Mystery Dungeon Maps
Loads maps from HuggingFace datasets and displays them in various formats
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
import argparse
import os

class MysteryDungeonMapViewer:
    def __init__(self):
        # ASCII characters for different tile types
        self.ascii_chars = {
            0: '#',  # Wall
            1: '.',  # Floor
            2: '~',  # Water (if you add it)
            3: '^',  # Lava (if you add it)
            4: '*',  # Treasure (if you add it)
        }
        
        # Colors for matplotlib visualization
        self.colors = {
            0: '#000000',  # Black walls
            1: '#8B4513',  # Brown floors
            2: '#0000FF',  # Blue water
            3: '#FF0000',  # Red lava
            4: '#FFD700',  # Gold treasure
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
    
    def extract_map_from_image(self, image_pil: Image.Image) -> np.ndarray:
        """Extract map array from PIL Image"""
        # Convert image back to numpy array
        img_array = np.array(image_pil)
        
        # If it's grayscale, use it directly
        if len(img_array.shape) == 2:
            # Normalize to 0-1 range
            normalized = img_array.astype(np.float32) / 255.0
            # Convert back to binary (0 or 1)
            map_array = (normalized > 0.5).astype(np.uint8)
        else:
            # If RGB, convert to grayscale first
            gray = np.mean(img_array, axis=2)
            normalized = gray.astype(np.float32) / 255.0
            map_array = (normalized > 0.5).astype(np.uint8)
        
        return map_array
    
    def display_ascii_map(self, map_array: np.ndarray, title: str = "Map", 
                         save_to_file: str = None):
        """Display map as ASCII art"""
        ascii_output = []
        ascii_output.append(f"\n{title}")
        ascii_output.append("=" * (map_array.shape[1] + 2))
        
        for row in map_array:
            ascii_row = ''.join([self.ascii_chars.get(tile, '?') for tile in row])
            ascii_output.append(f"|{ascii_row}|")
        
        ascii_output.append("=" * (map_array.shape[1] + 2))
        ascii_output.append(f"Size: {map_array.shape[1]}x{map_array.shape[0]}")
        ascii_output.append(f"Floor tiles: {np.sum(map_array == 1)}")
        ascii_output.append(f"Wall tiles: {np.sum(map_array == 0)}")
        
        # Print to console
        for line in ascii_output:
            print(line)
        
        # Save to file if requested
        if save_to_file:
            with open(save_to_file, 'w') as f:
                f.write('\n'.join(ascii_output))
            print(f"ASCII map saved to: {save_to_file}")
    
    def display_matplotlib_map(self, map_array: np.ndarray, title: str = "Map", 
                              figsize: Tuple[int, int] = (10, 8), save_to_file: str = None):
        """Display map using matplotlib"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create colored map
        colored_map = np.zeros((*map_array.shape, 3))
        for tile_type, color in self.colors.items():
            mask = map_array == tile_type
            colored_map[mask] = plt.matplotlib.colors.to_rgb(color)
        
        ax.imshow(colored_map)
        ax.set_title(title)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        
        # Add grid
        ax.set_xticks(range(0, map_array.shape[1], 5))
        ax.set_yticks(range(0, map_array.shape[0], 5))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to file if requested
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
        
        # Extract map array from image
        map_array = self.extract_map_from_image(row['image'])
        
        # Display map
        title = f"Map {index} (ID: {row.get('map_id', 'Unknown')})"
        
        # Create output directory if specified
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            base_filename = f"{output_dir}/map_{index:06d}"
        else:
            base_filename = None
        
        if display_format in ["ascii", "both"]:
            ascii_file = f"{base_filename}.txt" if base_filename else None
            self.display_ascii_map(map_array, title, ascii_file)
        
        if display_format in ["matplotlib", "both"]:
            img_file = f"{base_filename}.png" if base_filename else None
            self.display_matplotlib_map(map_array, title, save_to_file=img_file)
        
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
            map_array = self.extract_map_from_image(row['image'])
            
            # Create colored map
            colored_map = np.zeros((*map_array.shape, 3))
            for tile_type, color in self.colors.items():
                mask = map_array == tile_type
                colored_map[mask] = plt.matplotlib.colors.to_rgb(color)
            
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
    else:
        print("Please specify either --dataset or --local")
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