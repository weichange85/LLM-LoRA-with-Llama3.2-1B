from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"C:\Users\weich\.llama\checkpoints\Llama3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)