import torch
import swanlab
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from swanlab.integration.huggingface import SwanLabCallback

# 1. model and preprocess
MODEL_ID = "Qwen/Qwen3-0.6B"       # Base
DATASET_ID = "yahma/alpaca-cleaned" # Alpaca
OUTPUT_DIR = "./qwen3-alpaca-lora"  # output

# 2. Tokenizer and chat template：ChatML
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

# 3. dataset preprocess
# From Alpaca (instruction, input, output) to standard conversational (messages)
def preprocess_function(example):
    # Combined instruction and input as user content
    if example.get("input"):
        user_content = f"{example['instruction']}\n\nContext:\n{example['input']}"
    else:
        user_content = example['instruction']

    # return message
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]
    }

print("正在加载并处理数据...")
dataset = load_dataset(DATASET_ID, split="train")
train_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)

# 4. PEFT: LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,           # rank
    lora_alpha=32,  
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 5. SFTConfig
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=512,        # Maximum sequence length
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, 
    learning_rate=1e-4,
    num_train_epochs=1,         # Number of training epochs (1 for demo)
    logging_steps=10,
    fp16=False,
    bf16=True,
    packing=False,
    report_to="swanlab",
    run_name="Qwen3-0.6B-LoRA-Run1",
    dataset_text_field=None
)

# 7. Initialize Trainer
trainer = SFTTrainer(
    model=MODEL_ID,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    args=sft_config,
    peft_config=peft_config,
)

print("配置完成，开始训练...")
trainer.train()

print(f"训练完成！模型已保存至 {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)