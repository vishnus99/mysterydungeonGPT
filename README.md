# mysterydungeonGPT

Generate mystery dungeon-style maps from text descriptions using fine-tuned language models.

## Overview

mysterydungeonGPT is a complete pipeline for generating playable dungeon maps through the LoRA fine-tuned Qwen3-0.6B large language model. The project includes:

- **Dataset Generation**: Procedurally generate mystery dungeon maps with configurable parameters
- **Model Training**: Fine-tune Qwen3-0.6B using LoRA on generated map data
- **Inference**: Deploy the fine-tuned model on Modal for map generation
- **Web Game**: Play generated maps in a browser-based game

## Features

- **Text-to-Map Generation**: Describe a dungeon in natural language and get a playable map
- **Interactive Web Game**: Play generated maps with fog of war, enemy AI, and exploration
- **Map Visualization**: View and analyze maps using the built-in map viewer
- **Cloud Deployment**: Inference server deployed on Modal for easy access
- **Flexible Configuration**: Customize map size, room count, difficulty, and more!

## Installation

### Prerequisites

- Python 3.8+
- Modal account (for inference deployment)
- HuggingFace account (for dataset/model access)

## Quick Start

### 1. Generate a Map

The easiest way to generate a map:

```bash
python generate_map.py "Generate a medium difficulty dungeon with 8 rooms"
```

This will:
- Connect to the deployed inference server
- Generate a map from your prompt
- Save it to `web_game/maps/` and update the map index
- Make it immediately playable in the web game

### 2. Play the Web Game

```bash
cd web_game
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

### 3. View Maps

View generated maps using the map viewer:

```bash
# View all maps in the maps directory
python mapviewer.py --json-dir web_game/maps

# View a specific map
python mapviewer.py --json-file web_game/maps/map_generated_20260127_150652.json
```

## Usage

### Dataset Generation

Generate training data for fine-tuning:

```bash
python dataset/mapgenerator.py
```

This creates maps and uploads them to HuggingFace (default: `teamgas/mysterydungeondata`).

### Training

Train the model on Modal:

```bash
modal deploy modal_train.py
```

Or train locally (requires GPU):

```python
from mysterydungeonGPT.trainer import setup_and_train

setup_and_train(
    output_dir="./mysterydungeonGPT/trained_model/qwen3-lora-finetuned",
    batch_size=1,
    num_epochs=1,
    learning_rate=2e-4,
    max_context_length=2048
)
```

### Inference Deployment

Deploy the inference server on Modal:

```bash
modal deploy inference.py
```

The server will:
- Merge the LoRA adapter into the base model
- Launch SGLang server for high-performance inference
- Expose an OpenAI-compatible API endpoint

### Map Generation (Programmatic)

Use the `MapGenerator` class programmatically:

```python
from generate_map import MapGenerator

# Auto-discover URL from Modal
generator = MapGenerator()
result = generator.generate_and_save("Generate a hard dungeon with 12 rooms")

if result["success"]:
    print(f"Map saved: {result['output_path']}")
    print(f"Map ID: {result['map_id']}")
```

## Map Format

Maps are stored as JSON with the following structure:

```json
{
  "walkable_tiles": [[x1, y1], [x2, y2], ...],
  "player_spawn": [x, y],
  "stairs_spawn": [x, y],
  "width": 56,
  "height": 32,
  "difficulty": "medium",
  "enemies": [{"x": x, "y": y}, ...]
}
```

The coordinate-based format significantly reduces token count compared to full grid representation, and additional parameters can be added easily in order to retrain a new dataset with additional map elements.


## Requirements

See `requirements.txt` for full dependency list. Key dependencies:

- `torch` - PyTorch for model training/inference
- `transformers` - HuggingFace transformers library
- `peft` - Parameter-efficient fine-tuning (LoRA)
- `datasets` - HuggingFace datasets
- `modal` - Modal cloud platform
- `openai` - OpenAI-compatible API client
- `numpy`, `pillow` - Image/data processing


## Contributing

Contributions welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- Built on [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- Uses [SGLang](https://github.com/sgl-project/sglang) for high-performance inference
- Deployed on [Modal](https://modal.com)
- Dataset hosted on [HuggingFace](https://huggingface.co/teamgas/mysterydungeondata)
- Dataset generation was built upon map code from [SkyTemple](https://github.com/SkyTemple/dungeon-eos)
