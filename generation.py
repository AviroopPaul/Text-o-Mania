import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "What is natural language processing?"
max_length = 50
temperature = 3

input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids=input_ids, max_length=max_length, temperature=temperature)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print(generated_text)

text= generated_text.split('.')[:3]
str=" ".join(text)
print(str)