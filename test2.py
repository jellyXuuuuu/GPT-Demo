import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel

# # load online
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load from local files.
load_path = '/home/flyslice/xy/gpt/huggingface/'
tokenizer = GPT2Tokenizer.from_pretrained(load_path)
# model = GPT2Config.from_json_file(load_path + "config.json")

# # Add special tokens
# special_tokens = {"pad_token":"<|pad|>"}
# tokenizer.add_special_tokens(special_tokens)

# text = 'Who was Jim Henson ? Jim Henson was a'
text = 'The White man worked as a plumber at the'

indexed_tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([indexed_tokens])

model = GPT2LMHeadModel.from_pretrained(load_path)
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

predicted_index = torch.argmax(predictions[0, -1, :]).item()
predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

print(predicted_text)
