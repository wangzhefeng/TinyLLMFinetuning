# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hf_finetuning.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-07-08
# * Version     : 1.0.070816
# * Description : description
# * Link        : https://www.philschmid.de/fine-tune-llms-in-2024-with-trl#1-define-our-use-case
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from trl import setup_chat_format, SFTTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM

from utils.online import hf_hub_login, wandb_login

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# data
# ------------------------------
# convert dataset to OAI messages
system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message.format(schema=sample["context"])},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
    }


def get_dataset():
    # data path
    data_path = Path(ROOT).joinpath("./dataset/SQL-Create-Context-SFT")
    train_path = data_path.joinpath("train_dataset.json")
    test_path = data_path.joinpath("test_dataset.json")
    if not train_path.exists() and not test_path.exists():
        # load dataset from the huggingface hub
        dataset = load_dataset(
            "b-mc2/sql-create-context", 
            split="train", 
            cache_dir=data_path,
        )
        dataset = dataset.shuffle().select(range(12500))
        # Convert dataset to OAI messages
        dataset = dataset.map(
            create_conversation, 
            remove_columns=dataset.features, 
            batched=False,
        )
        # Split dataset into 10,000 training samples and 2,500 test samples
        dataset = dataset.train_test_split(test_size=2500/12500) 
        # Save datasets to disk
        dataset["train"].to_json(train_path, orient="records")
        dataset["test"].to_json(test_path, orient="records")
        # print(dataset.keys())
        # print(dataset["train"][345]["messages"])

    # TODO load dataset
    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    # test_dataset = load_dataset("json", data_files=str(test_path), split="test")
    
    return train_dataset#, test_dataset


# ------------------------------
# model
# ------------------------------
# Hugging Face model id
model_id = "codellama/CodeLlama-7b-hf"  # or 'mistralai/Mistral-7B-v1.0'

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_type=torch.bfloat16,
    quantization_config=bnb_config,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "right"

# set chat template to OAI chatML, remove if you start from a fine-tuned model
model, tokenizer = setup_chat_format(model, tokenizer)

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
    r=256,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# training config
args = TrainingArguments(
    output_dir="code-llama-7b-text-to-sql", # directory to save and repository id
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=3,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)




# 测试代码 main 函数
def main():
    # env
    # hugging face login
    hf_hub_login.login()

    # wandb login
    wandb_login.login()

    # dataset
    train_dataset = get_dataset()

    # Trainer
    # max sequence length for model and packaging of the dataset
    max_seq_length = 3072
    trainer = SFTTrainer(
        model=model, 
        args=args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )

    # Model training
    trainer.train()
    # Model saving
    trainer.save_model()
    
    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()

    # Merge peft and base model
    # Load PEFT model on CPU
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    # Merge LoRA and base model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(
        args.output_dir,safe_serialization=True, 
        max_shard_size="2GB"
    )

if __name__ == "__main__":
    main()
