import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 1. Configuration
BASE_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
LORA_PATH = "./qwen3-alpaca-lora-text"
MERGED_DIR = "./merged_qwen3_0.6B_text"

# 2. Load Base Model
print(f"Loading Base Model: {BASE_MODEL_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

# 3. Load and Merge LoRA
print(f"Loading LoRA adapter from: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print("Merging weights (Merge and Unload)...")
model = model.merge_and_unload()

# 4. Save Merged Model
print(f"Saving merged model to: {MERGED_DIR}")
model.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)

print("-" * 30)
print(f"transformers chat {MERGED_DIR} --trust_remote_code")
print("-" * 30)