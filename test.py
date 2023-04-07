with open("sentences.txt", "r") as f:
    sentences = f.readlines()

sentences = [sentence.strip() for sentence in sentences]

print(sentences[1:10])

WINDOW_SIZE = 3
inputs = []
outputs = []

for sequence in sequences:
    for i in range(WINDOW_SIZE, len(sequence)):
        # Extract the input sequence and output word
        input_seq = sequence[i-WINDOW_SIZE:i]
        output_word = sequence[i]
        
        # Add the input sequence and output word to the respective lists
        inputs.append(input_seq)
        outputs.append(output_word)