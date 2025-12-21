import os
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import swanlab

warnings.filterwarnings("ignore")

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DATASET_ID = "yahma/alpaca-cleaned"
OUTPUT_DIR = "./qwen2.5-0.5B-alpaca-lora"
os.environ["SWANLAB_PROJECT"] = "Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512

im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

def create_text_column(example):
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()

    if inp:
        user = f"{inst}\nContext:\n{inp}"
    else:
        user = inst

    text = (
        "<|im_start|>user\n"
        f"{user}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{out}\n"
        "<|im_end|>\n"
    )
    return {"text": text}

dataset = load_dataset(DATASET_ID, split="train")
train_dataset = dataset.map(create_text_column, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

trainable_special_ids = tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    trainable_token_indices=trainable_special_ids,
)

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=512,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    fp16=False,
    bf16=True,
    packing=False,
    report_to="swanlab",
    dataset_text_field="text",
)

if model.generation_config is None:
    model.generation_config = GenerationConfig.from_pretrained(MODEL_ID)

model.generation_config.eos_token_id = im_end_id
model.generation_config.pad_token_id = im_end_id

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=sft_config,
    peft_config=peft_config,
)

trainer.train()

print(f"保存模型至 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("训练完成")
