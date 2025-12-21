import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuration
BASE_MODEL_ID = "./model_hub/Qwen3-0.6B-Base" # 建议用 HuggingFace ID，或者你本地确定的路径
# Path to saved LoRA adapter (你最新训练的那个)
LORA_PATH = "./qwen3-alpaca-lora-special-tokens"

# 2. Load Models
print(f"Loading Base Model from {BASE_MODEL_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

print(f"Loading LoRA weights: {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# ==========================================
# 🔥 关键修改 1：获取正确的停止符 ID
# ==========================================
# 你的模型训练目标是输出 <|im_end|> 来停止，所以必须告诉脚本它的 ID 是多少
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"🛑 设定停止符 ID: {im_end_id} (<|im_end|>)")

# 3. Prepare Input
prompt = "Give me a plan to study machine learning."
print(f"\n[User Instruction]: {prompt}\n")

# ==========================================
# 🔥 关键修改 2：手动拼接格式 (对齐训练代码)
# ==========================================
# 训练时你没有用 chat_template，而是手动拼的字符串。
# 推理时必须完全一致！否则模型会懵。
text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# 4. Generate Response
inputs = tokenizer([text], return_tensors="pt").to(base_model.device)

print("[AI is thinking...]")
generated_ids = model.generate(
    inputs.input_ids,
    max_new_tokens=512,
    
    # 推荐参数：测试时用 Greedy Search (do_sample=False) 看最稳定的能力
    do_sample=False, 
    # temperature=0.7, # 如果打开 do_sample=True 再用这些
    # top_p=0.9,
    
    # ==========================================
    # 🔥 关键修改 3：强制刹车
    # ==========================================
    eos_token_id=im_end_id,  # 见到 <|im_end|> 立即停止
    pad_token_id=im_end_id   # 防止报错
)

# 5. Decode Output
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

# ==========================================
# 🔥 关键修改 4：显示特殊字符
# ==========================================
# 暂时设为 False，让我们亲眼确认模型是不是输出了 <|im_end|>
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

print("-" * 20)
print(response)
print("-" * 20)

# 自动验证
if "<|im_end|>" in response:
    print("✅ 测试通过！模型成功学会了刹车！")
else:
    print("⚠️ 依然没停住，请检查训练数据结尾。")