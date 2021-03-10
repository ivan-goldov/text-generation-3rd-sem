import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
model = AutoModelForCausalLM.from_pretrained('distilgpt2')

model.load_state_dict(torch.load('./gpt2.bin'))

while i:=input():
    out = model.generate(tokenizer.encode(i, return_tensors='pt'), max_length=10)
    print(tokenizer.decode(out[0]))
