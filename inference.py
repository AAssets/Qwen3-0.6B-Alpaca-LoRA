import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuration
BASE_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
# Path to saved LoRA adapter
LORA_PATH = "./qwen3-alpaca-lora"

# 2. Load Models
print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

print(f"Loading LoRA weights: {LORA_PATH}")
# Load the LoRA adapter onto the base model
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 3. Prepare Input
prompt = "Give me a plan to study machine learning."

print(f"\n[User Instruction]: {prompt}\n")

messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Generate Response
inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

print("[AI is thinking...]")
generated_ids = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

# 5. Decode Output
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 20)
print(response)
print("-" * 20)