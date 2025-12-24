import torch
from torch.utils.data import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from mysterydungeonGPT.helpers import format_map_for_training, train, evaluate


class MapDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_context_length = 6144):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length

    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        prompt, json_output = format_map_for_training(example)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json_output}
        ] 
        
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False
        )
        user_length = user_only['input_ids'].shape[1]

        full_tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            max_length=self.max_context_length,
            truncation=True,
            padding="max_length"
        )
        input_ids = full_tokenized['input_ids'].squeeze(0)
        attention_mask = full_tokenized['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[:user_length] = -100

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padding_mask = (input_ids == pad_token_id)
        labels[padding_mask] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def setup_and_train(
    output_dir="./mysterydungeonGPT/trained_model/qwen3-lora-finetuned",
    batch_size=4,
    num_epochs=1,
    learning_rate=2e-4,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    eval_steps=50,
    save_steps=100,
    max_context_length=6144,
    device_map="auto"
):
    """
    Setup model, datasets, and run training.
    
    Args:
        output_dir: Directory to save checkpoints and final model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        eval_steps: Evaluate every N steps (None = end of epoch)
        save_steps: Save checkpoint every N steps (None = end of epoch)
        max_context_length: Maximum sequence length
        device_map: Device mapping for model loading ("auto", "cuda", "cpu")
    
    Returns:
        Trained model
    """
    #tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        device_map=device_map
    )

    #dataset
    dataset = load_dataset("teamgas/mysterydungeondata")

    print("Available splits:", list(dataset.keys()))

    # If validation split doesn't exist, create it from train
    if 'validation' not in dataset:
        # Split the train dataset into train and validation
        split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
        dataset['train'] = split_dataset['train']
        dataset['validation'] = split_dataset['test']
        print(f"Created validation split: {len(dataset['validation'])} samples")


    #train and val datasets
    train_dataset = MapDataset(
        hf_dataset=dataset['train'],
        tokenizer=tokenizer,
        max_context_length=max_context_length
    )

    val_dataset = MapDataset(
        hf_dataset=dataset['validation'],
        tokenizer=tokenizer,
        max_context_length=max_context_length
    )

    #lora config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, #Rank of low rank adaptation
        lora_alpha=16, #Scaling parameter (usually 2x rank)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], #Which modules to apply LoRA to (attention and MoE layers)
        lora_dropout=0.1,
        bias="none",
        inference_mode=False
    )

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #training
    trained_model = train(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lora_config=lora_config,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir
    )

    return trained_model

if __name__ == "__main__":
    setup_and_train()