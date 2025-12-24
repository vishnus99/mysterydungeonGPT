import modal

app = modal.App("mystery-dungeon-trainer")

image = (
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "tqdm",
        "numpy",
        "pillow"
    )
    .apt_install("git")
    .run_commands("git clone https://github.com/vishnus99/mysterydungeonGPT.git /root/mysterydungeonGPT")
)

volume = modal.Volume.from_name("mystery-dungeon-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/checkpoints": volume},
    timeout=3600
)

def train_model():
    import sys
    import os

    sys.path.insert(0, "/root/mysterydungeonGPT")

    from mysterydungeonGPT.trainer import setup_and_train

    output_dir = "/checkpoints/qwen3-lora-finetuned"
    os.makedirs(output_dir, exist_ok=True)

    trained_model = setup_and_train(
        output_dir=output_dir,
        batch_size=4,
        num_epochs=1,
        learning_rate=2e-4,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        eval_steps=50,
        save_steps=100,
        max_context_length=6144,
        device_map="auto"
    )

    final_model_path = f"{output_dir}/final_model"
    trained_model.save_pretrained(final_model_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.save_pretrained(final_model_path)

    volume.commit()

    return final_model_path

@app.local_entrypoint()
def main():
    result = train_model.remote()
    print(f"Training completed: {result}")
