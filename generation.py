import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate some text
while True:
    prompt = input("Enter prompt: ")
    if prompt.strip() == '':
        print("Please enter a valid prompt.")
        continue
    max_length = input("Enter maximum length of generated text: ")
    try:
        max_length = int(max_length)
    except ValueError:
        print("Please enter a valid integer for the maximum length.")
        continue
    top_p = input("Enter top_p value for sampling (default is 0.95): ")
    top_k = input("Enter top_k value for sampling (default is 50): ")
    try:
        if top_p.strip() == '':
            top_p = 0.95
        else:
            top_p = float(top_p)
        if top_k.strip() == '':
            top_k = 50
        else:
            top_k = int(top_k)
    except ValueError:
        print("Please enter valid numbers for the sampling parameters.")
        continue
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, do_sample=True, top_p=top_p, top_k=top_k)
    # Print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated text: {generated_text}")
    break