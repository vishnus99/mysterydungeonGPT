import json
import random
import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import get_peft_model
try:
    import wandb
    WANDB_AVAILABLE = True
except ModuleNotFoundError:
    WANDB_AVAILABLE = False
    wandb = None

def extract_json_from_text(text):
    """
    Extract JSON object from text, handling incomplete JSON.
    
    Args:
        text: Text containing JSON object
    
    Returns:
        Parsed JSON dictionary or None if extraction fails
    """
    # Try to find JSON object
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        print("No JSON found in generated content")
        print(f"First 500 chars: {text[:500]}")
        return None
    
    json_str = json_match.group()
    
    # Try to parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"JSON string length: {len(json_str)}")
        print(f"Last 200 chars: {json_str[-200:]}")
        
        # Try to fix common issues
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        try:
            return json.loads(json_str)
        except:
            print("Could not fix JSON")
            return None

def coordinates_to_grid(walkable_coords, width=56, height=32):
    """
    Convert list of walkable coordinates back to full grid array.
    
    Args:
        walkable_coords: List of [x, y] coordinates representing walkable tiles
        width: Grid width (default 32)
        height: Grid height (default 32)
    
    Returns:
        2D numpy array where 0=wall, 1=floor
    """
    grid = np.zeros((height, width), dtype=np.uint8)
    
    for coord in walkable_coords:
        x, y = coord
        if 0 <= x < width and 0 <= y < height:
            grid[y, x] = 1  # Mark as walkable floor
    
    return grid

def format_map_for_training(map):
    room_count = map['room_count']
    complexity = map['complexity']
    difficulty = map['difficulty']

    enemies = json.loads(map['enemies'])
    gen_params = json.loads(map['generation_params'])

    # Get map_array from dataset - stored as JSON string (source of truth)
    map_array_raw = map.get('map_array')
    
    if map_array_raw is None:
        raise ValueError("map_array is missing from dataset entry")
    
    map_width = map.get('width', 32) 
    map_height = map.get('height', 32)
    
    # Parse JSON string to numpy array
    try:
        parsed = json.loads(map_array_raw)
        map_array = np.array(parsed, dtype=np.uint8)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse map_array JSON: {e}")
    
    # Validate shape
    if map_array.shape != (map_height, map_width):
        raise ValueError(f"map_array shape {map_array.shape} does not match expected ({map_height}, {map_width})")

    player_x = map['player_spawn_x']
    player_y = map['player_spawn_y']
    stairs_x = map['stairs_spawn_x']
    stairs_y = map['stairs_spawn_y']

    templates = [
        # Direct/Simple
        "Generate a {difficulty} difficulty dungeon with {room_count} rooms",
        "Create a {difficulty} dungeon containing {room_count} rooms",
        "Design a {difficulty} level dungeon with {room_count} rooms",
        
        # With Complexity
        "Generate a {difficulty} difficulty dungeon with {room_count} rooms and complexity {complexity}",
        "Create a dungeon map: {difficulty} difficulty, {room_count} rooms, complexity {complexity}",
        "Design a {difficulty} dungeon with {room_count} rooms at complexity level {complexity}",
        
        # Structured/Technical
        "Generate dungeon: difficulty={difficulty}, room_count={room_count}, complexity={complexity}",
        "Create dungeon map with parameters: difficulty={difficulty}, rooms={room_count}, complexity={complexity}",
        "Dungeon specifications: {difficulty} difficulty, {room_count} rooms, complexity {complexity}",
        
        # Conversational
        "I need a {difficulty} dungeon with {room_count} rooms",
        "Please generate a {difficulty} difficulty dungeon containing {room_count} rooms",
        "Can you create a {difficulty} dungeon map with {room_count} rooms?",
        
        # Descriptive
        "Generate a {difficulty} difficulty dungeon featuring {room_count} rooms and complexity {complexity}",
        "Create a {difficulty} level dungeon map with {room_count} rooms, complexity set to {complexity}",
        "Design a {difficulty} dungeon containing {room_count} rooms with a complexity of {complexity}",
        
        # Game Context
        "For a roguelike game, generate a {difficulty} dungeon with {room_count} rooms",
        "Create a playable {difficulty} dungeon map with {room_count} rooms and complexity {complexity}",
        "Generate a {difficulty} dungeon layout: {room_count} rooms, complexity {complexity}",
        
        # Varied Phrasing
        "Build a {difficulty} dungeon consisting of {room_count} rooms",
        "Produce a {difficulty} difficulty dungeon map with {room_count} rooms",
        "Make a {difficulty} dungeon with {room_count} rooms, complexity {complexity}",
        
        # More Natural
        "A {difficulty} dungeon with {room_count} rooms, please",
        "Generate me a {difficulty} difficulty dungeon that has {room_count} rooms",
        "I want a {difficulty} dungeon map with {room_count} rooms and complexity {complexity}",
        
        # Compact
        "{difficulty} dungeon, {room_count} rooms, complexity {complexity}",
        "Dungeon: {difficulty}, {room_count} rooms, complexity {complexity}",
        "Map: {difficulty} difficulty, {room_count} rooms, {complexity} complexity",
    ]

    prompt = random.choice(templates).format(
        difficulty = difficulty,
        room_count = room_count,
        complexity = round(complexity, 2)
    )

    # Convert map_array to numpy if needed and normalize spawn points to floors
    if isinstance(map_array, np.ndarray):
        # Convert spawn points (2, 3) to floors (1) for coordinate extraction
        map_array = np.where((map_array == 2) | (map_array == 3), 1, map_array)
    elif isinstance(map_array, list):
        map_array = np.array(map_array)
        map_array = np.where((map_array == 2) | (map_array == 3), 1, map_array)
    else:
        map_array = np.array(map_array)
        map_array = np.where((map_array == 2) | (map_array == 3), 1, map_array)

    # Extract only walkable coordinates (where value == 1)
    walkable_coords = []
    for y in range(map_height):
        for x in range(map_width):
            if map_array[y, x] == 1:  # Walkable floor tile
                walkable_coords.append([x, y])  # Store as [x, y] format
    
    # Ensure spawn points are on floor tiles and in walkable_coords
    # If spawn coordinates are not on floors, find nearest floor tile
    walkable_coords_set = set(tuple(c) for c in walkable_coords)
    
    # Validate and fix player spawn (safety check - should be rare after map generator fix)
    if 0 <= player_x < map_width and 0 <= player_y < map_height:
        if map_array[player_y, player_x] != 1:
            # Spawn is not on a floor tile - find nearest floor
            min_dist = float('inf')
            nearest_floor = None
            for x, y in walkable_coords:
                dist = abs(x - player_x) + abs(y - player_y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_floor = [x, y]
            if nearest_floor:
                player_x, player_y = nearest_floor
        
        # Ensure spawn coordinate is in walkable_coords
        if (player_x, player_y) not in walkable_coords_set:
            walkable_coords.append([player_x, player_y])
            walkable_coords_set.add((player_x, player_y))
    
    # Validate and fix stairs spawn (safety check - should be rare after map generator fix)
    if 0 <= stairs_x < map_width and 0 <= stairs_y < map_height:
        if map_array[stairs_y, stairs_x] != 1:
            # Spawn is not on a floor tile - find nearest floor
            min_dist = float('inf')
            nearest_floor = None
            for x, y in walkable_coords:
                dist = abs(x - stairs_x) + abs(y - stairs_y)
                if dist < min_dist:
                    min_dist = dist
                    nearest_floor = [x, y]
            if nearest_floor:
                stairs_x, stairs_y = nearest_floor
        
        # Ensure spawn coordinate is in walkable_coords
        if (stairs_x, stairs_y) not in walkable_coords_set:
            walkable_coords.append([stairs_x, stairs_y])
            walkable_coords_set.add((stairs_x, stairs_y))
    
    # Sort coordinates for consistency (row-major order: sort by y first, then x)
    walkable_coords.sort(key=lambda coord: (coord[1], coord[0]))

    # Create coordinate-based JSON format
    game_dict = {
        'walkable_tiles': walkable_coords,  # List of [x, y] coordinates
        'player_spawn': [player_x, player_y],
        'stairs_spawn': [stairs_x, stairs_y],
            'width': 56,
            'height': 32,
        'difficulty': difficulty,
        'enemies': enemies
    }

    json_string = json.dumps(game_dict)

    return prompt, json_string

def train(
    model,
    tokenizer,
    batch_size,
    num_epochs,
    learning_rate,
    device,
    train_dataset,
    val_dataset,
    lora_config,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    eval_steps=None,
    save_steps=None,
    output_dir=None
):
    """ Main Training Loop

    Args:
        model (AutoModelForCausalLM): Model to train
        tokenizer (AutoTokenizer): Tokenizer for the model
        batch_size (int): Batch size
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate
        device (torch.device): Device to train on
        train_dataset: Training dataset
        val_dataset: Validation dataset
        lora_config: LoRA configuration
        gradient_accumulation_steps (int): Number of steps to accumulate gradients
        max_grad_norm (float): Maximum gradient norm for clipping
        eval_steps (int): Evaluate every N steps (None = end of epoch)
        save_steps (int): Save checkpoint every N steps (None = end of epoch)
        output_dir (str): Directory to save checkpoints
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize Weights & Biases with error handling
    wandb_initialized = False
    if wandb is None:
        print("wandb not available, logging disabled")
    else: 
        try:
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            if wandb_api_key:
                wandb.init(
                    project="mystery-dungeon-gpt",
                    name=f"qwen3-0.6b-lora-32x32",
                    config={
                        "model": "Qwen3-0.6B",
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "max_grad_norm": max_grad_norm,
                        "max_context_length": train_dataset.max_context_length if hasattr(train_dataset, 'max_context_length') else None,
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset)
                    }
                )
                wandb_initialized = True
                print("Weights & Biases initialized successfully")
            else:
                print("WANDB_API_KEY not found in environment - wandb logging disabled")
                wandb.init(mode="disabled")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            print("Continuing training without wandb logging")
            wandb.init(mode="disabled")

    num_training_steps = len(train_dataloader) * num_epochs
    
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if global_step % 10 == 0 and wandb_initialized:  # Log every 10 steps to avoid too much logging
                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/step": global_step,
                    "train/epoch": epoch + 1,
                })

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=max_grad_norm
                )

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if eval_steps and global_step % eval_steps == 0:
                    val_loss = evaluate(model, val_dataloader, device)
                    model.train()
                    
                    # Log validation loss to wandb
                    if wandb_initialized:
                        wandb.log({
                            "val/loss": val_loss,
                            "val/step": global_step,
                        })
                
                if save_steps and global_step % save_steps == 0:
                    if output_dir:
                        checkpoint_path = f"{output_dir}/checkpoint-{global_step}"
                        model.save_pretrained(checkpoint_path)
                        print(f"Saved checkpoint to {checkpoint_path}")
        
            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
    
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

        val_loss = evaluate(model, val_dataloader, device)

        if wandb_initialized:
            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": val_loss,
                "epoch": epoch + 1
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if output_dir:
                best_model_path = f"{output_dir}/best_model"
                model.save_pretrained(best_model_path)
                print(f"Saved best model (val_loss={val_loss:.4f}) to {best_model_path}")
    
    # Finish wandb run
    if wandb_initialized:
        wandb.finish()
        print("Weights & Biases run completed")
    
    return model

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches
    return avg_loss