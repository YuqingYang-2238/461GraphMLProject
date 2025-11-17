#pip install "transformers>=4.40.0" peft datasets accelerate bitsandbytes pandas

import os
import torch
import pandas as pd
from dataclasses import dataclass

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from huggingface_hub import login
login(token="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")
# --------- CONFIG ---------
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"   # larger open-source Qwen
Train_DATA_PATH = "./data/BBBP_train_data.csv" 
Val_DATA_PATH = "./data/BBBP_val_data.csv"
OUTPUT_DIR = "./qwen2_7B_bbbp_lora"
MAX_LENGTH = 1024
BATCH_SIZE = 2
NUM_EPOCHS = 3
LR = 1e-4
WARMUP_RATIO = 0.05
USE_8BIT = True                           
SEED = 42



def load_data(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Validate data
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Train label distribution: {train_df['Label'].value_counts().to_dict()}")
    print(f"Max SMILES length: {train_df['SMILES'].str.len().max()}")
    

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    return DatasetDict(train=train_dataset, validation=val_dataset)

def build_prompt(smiles, label):
    system_msg = "You are an expert in molecular chemistry analyzing blood-brain barrier penetration."
    
    user_msg = f"""Analyze this molecule and predict Blood-Brain Barrier (BBB) penetration.

                   SMILES: {smiles}

                   Respond with JSON: {{"answer": 0}} for non-BBBpenetrant or {{"answer": 1}} for penetrant.
                   You must return ONLY the JSON object with no additional text or explanation.
		"""
    
    prompt_text = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>\n"

    if label is None:
        return prompt_text, None
    else:
        target_text = f'{{"answer": {int(label)}}}'  # Match the JSON format!
        return prompt_text, target_text


def make_preprocess_fn(tokenizer):

    def preprocess(example):
        prompt, target = build_prompt(example["SMILES"], example["Label"])

        # tokenize prompt and full text (prompt + answer)
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
        )
        full_text = prompt + target +tokenizer.eos_token 
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            add_special_tokens=True,
            padding="max_length",  # ADD THIS
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
    
        # We want loss ONLY on the target tokens (after the prompt).
        labels = [-100] * len(input_ids)
        prompt_len = len(prompt_tokens["input_ids"])
        labels[prompt_len:] = input_ids[prompt_len:]
        
        # Also set padding tokens in labels to -100
        labels = [
            -100 if attention_mask[i] == 0 else labels[i] 
            for i in range(len(labels))
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return preprocess

from transformers import BitsAndBytesConfig

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if USE_8BIT:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    if USE_8BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. load data
    dataset = load_data(Train_DATA_PATH, Val_DATA_PATH)

    # 2. model + tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # 3. preprocessing
    preprocess_fn = make_preprocess_fn(tokenizer)
    tokenized_train = dataset["train"].map(
        preprocess_fn,
        remove_columns=dataset["train"].column_names,
    )

    print(tokenized_train[0].keys())
    print(len(tokenized_train[0]["input_ids"]), len(tokenized_train[0]["labels"]))
    tokenized_val = dataset["validation"].map(
        preprocess_fn,
        remove_columns=dataset["validation"].column_names,
    )

    # 4. data collator
    data_collator = default_data_collator

    # 5. training args
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",    # Changed to steps
    eval_steps=200,                  # Evaluate every 200 steps
    save_strategy="steps",           # Changed to steps
    save_steps=200,                  # Save every 200 steps
    logging_strategy="steps",
    logging_steps=50,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    bf16=torch.cuda.is_available(),
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=SEED,
)
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training finished, model saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
