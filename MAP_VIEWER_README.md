# Mystery Dungeon Map Viewer

A comprehensive tool for viewing and exporting Mystery Dungeon maps from HuggingFace datasets.

## Features

- **Load maps** from HuggingFace datasets or local parquet files
- **Display maps** as ASCII art or matplotlib visualizations
- **Export maps** in multiple formats:
  - ASCII text files (.txt)
  - PNG images (.png)
  - NumPy arrays (.npy)
  - CSV files (.csv)
  - Binary files for ROM hacking (.bin)
- **Search and filter** maps by criteria
- **Interactive mode** for browsing maps
- **Batch export** all maps in a dataset

## Installation

Make sure you have the required dependencies:

```bash
pip install numpy pandas datasets pillow matplotlib seaborn
```

## Usage

### Command Line Interface

#### View a single map from HuggingFace:
```bash
python mapviewer.py --dataset "vishnusm/mysterydungeonmaps" --index 0 --format both
```

#### View a map and save to files:
```bash
python mapviewer.py --dataset "vishnusm/mysterydungeonmaps" --index 5 --output-dir "./my_maps"
```

#### Export all maps in a dataset:
```bash
python mapviewer.py --dataset "vishnusm/mysterydungeonmaps" --export-all --output-dir "./all_maps"
```

#### Interactive mode:
```bash
python mapviewer.py --dataset "vishnusm/mysterydungeonmaps" --interactive
```

#### View local parquet file:
```bash
python mapviewer.py --local "path/to/dataset.parquet" --index 0
```

### Python Script Usage

```python
from mapviewer import MysteryDungeonMapViewer

# Create viewer
viewer = MysteryDungeonMapViewer()

# Load dataset
df = viewer.load_dataset_from_huggingface("vishnusm/mysterydungeonmaps")

# View a single map
viewer.view_single_map(df, index=0, display_format="both")

# Export a map to files
viewer.view_single_map(df, index=0, output_dir="./exported_maps")

# Search for maps
indices = viewer.search_maps(df, room_count=(5, 8))
viewer.view_multiple_maps(df, indices)

# Export all maps
viewer.export_all_maps(df, "./all_maps_export")
```

## Output Files

When you specify an `--output-dir`, the following files are created for each map:

- `map_XXXXXX.txt` - ASCII representation
- `map_XXXXXX.png` - High-resolution PNG image
- `map_XXXXXX.npy` - NumPy array file
- `map_XXXXXX.csv` - CSV format
- `map_XXXXXX_scaled.png` - Scaled PNG for better visibility
- `map_XXXXXX.bin` - Binary format for ROM hacking
- `map_XXXXXX_metadata.txt` - Map metadata

## Map Format

The maps are stored as numpy arrays where:
- `0` = Wall/blocked tile (displayed as `#`)
- `1` = Floor/passable tile (displayed as `.`)

## ROM Hacking Integration

The `.bin` files are formatted for Pokemon Red Rescue Team ROM hacking:
- Wall tiles: `0x00`
- Floor tiles: `0x01`

## Examples

See `example_map_viewer.py` for complete usage examples.

## Interactive Commands

When using `--interactive` mode:
- `view <index>` - View map at index
- `random` - View random map
- `search <room_count>` - Search for maps with specific room count
- `stats` - Show dataset statistics
- `quit` - Exit
