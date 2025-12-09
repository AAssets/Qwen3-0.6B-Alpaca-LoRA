import torch
import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import swanlab

MODEL_ID = "Qwen/Qwen3-0.6B-Base" 
DATASET_ID = "yahma/alpaca-cleaned"
OUTPUT_DIR = "./qwen3-alpaca-lora-text"

# SwanLab
os.environ["SWANLAB_PROJECT"] = "Qwen0.6B-RawText"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 512

def create_text_column(example):
    
    inst = example['instruction']
    inp = example['input']
    out = example['output']
    
    if inp:
        text = f"<|im_start|>user\n{inst}\nContext:\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>\n"
    else:
        text = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>\n"
        
    # 返回一个包含 'text' 键的字典
    return {"text": text}

print("正在加载数据...")
dataset = load_dataset(DATASET_ID, split="train")

train_dataset = dataset.map(create_text_column, remove_columns=dataset.column_names)

# 打印一条看看对不对
print("="*20 + " 数据样例 " + "="*20)
print(train_dataset[0]["text"])
print("="*60)

# LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, 
    lora_alpha=32, 
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# SFTConfig
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=512,
    
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=3,
    logging_steps=10,
    fp16=False, bf16=True,
    packing=False,
    report_to="swanlab",
    run_name="Qwen0.6B-RawText-Run1",
    
    dataset_text_field="text"
)

print(f"正在加载模型: {MODEL_ID} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=sft_config,
    peft_config=peft_config,
)

print("配置完成，开始 Text 模式训练...")
trainer.train()

print(f"训练完成！模型已保存至 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)