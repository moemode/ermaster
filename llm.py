from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM(
    "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
)