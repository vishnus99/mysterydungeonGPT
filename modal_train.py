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
        "pillow",
        "wandb"
    )
    .apt_install("git")
    .add_local_python_source("mysterydungeonGPT")
)

volume = modal.Volume.from_name("mystery-dungeon-checkpoints", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    volumes={"/checkpoints": volume},
    timeout=86400,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb")
    ]
)

def train_model():
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    from mysterydungeonGPT.trainer import setup_and_train

    output_dir = "/checkpoints/qwen3-lora-finetuned"
    os.makedirs(output_dir, exist_ok=True)

    trained_model = setup_and_train(
        output_dir=output_dir,
        batch_size=1,
        num_epochs=1,
        learning_rate=2e-4,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        eval_steps=50,
        save_steps=100,
        max_context_length=2048,
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
