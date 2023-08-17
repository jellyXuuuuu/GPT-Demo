import torch
from transformers import GPT2Tokenizer, GPT2Model, \
GPT2Config, GPT2LMHeadModel, AutoTokenizer
from itertools import chain
from transformers.utils import logging

# logging.set_verbosity_info()

# # load online
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load from local files.
load_path = '/home/flyslice/xy/gpt/huggingface/'
# tokenizer = GPT2Tokenizer.from_pretrained(load_path, padding_size="left")
tokenizer = AutoTokenizer.from_pretrained(load_path, padding_size="left")


# model = GPT2Config.from_json_file(load_path + "config.json")

# Add special tokens
special_tokens = {"pad_token":"<|pad|>"}
tokenizer.add_special_tokens(special_tokens)

# Encode text.
text = "I'm reading a book"
# base_tokenizer
text_ids = tokenizer.encode(text, return_tensors='pt')

# model.resize_token_embeddings(len(tokenizer))
# model.tie_weights()



user_1 = "Does money buy happiness?"
bot_1 = "Depends how much money you spend on it."
user_2 = "What is the best way to buy happiness?"
bot_2 = "You just have to be a millionaire by your early 20s, \
        then you can be happy."
user_3 = "This is so difficult!"
bot_3 = "You have no idea how hard it is to be a millionaire \
        and happy. There is a reason the rich have a lot of money"



bos, eos, pad, sep = tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id
user_tti, bot_tti, pad_tti = 0, 1, 2
context_list = [user_1, bot_1, user_2, bot_2, user_3, bot_3]

input_ids = [[sep] + tokenizer.encode(utter) for utter in context_list]
token_type_ids = [[user_tti] * len(tokenizer.encode(utter)) if i % 2 ==0 else [bot_tti] * len(tokenizer.encode(utter)) for i, utter in enumerate(context_list)]
lm_labels = [([pad] * (len(list(chain(*input_ids))) - len(tokenizer.encode(bot_3)) - 1))] + ([tokenizer.encode(bot_3)]) + ([eos])

# print(lm_labels)
new_lm_labels = []
for i in range(len(lm_labels)-1):
    for j in range(len(lm_labels[i])):
        new_lm_labels.append(lm_labels[i][j])

    if i == len(lm_labels)-1:
        new_lm_labels.append(lm_labels[i])

# print(new_lm_labels)

new_input_ids = list(chain(*input_ids))
for x in range(len(new_input_ids)):
    if new_input_ids[x] == None:
        new_input_ids[x] = 0
# print(new_input_ids)

new_token_type_ids = list(chain(*token_type_ids))
for i in range(6):
    new_token_type_ids.append(1)

# print(new_token_type_ids)

data = {
    "input_ids": new_input_ids, #list(chain(*input_ids)),
    "token_type_ids": new_token_type_ids, #list(chain(*token_type_ids))
    "lm_labels": new_lm_labels,  #list(chain(*lm_labels))
}




# print(input_ids)
# print(list(chain(*input_ids)))

position_ids = torch.arange(len(new_input_ids), dtype=torch.long) #input_ids
input_ids = torch.tensor(data["input_ids"], dtype=torch.long)
token_type_ids = torch.tensor(data["token_type_ids"], dtype=torch.long)
# lm_labels = torch.tensor(data["lm_labels"], dtype=torch.long)

# print(position_ids)
# print(position_ids.shape)

# When training.
# remove outputs? 2 values returned
losses, logits = model(
    input_ids=input_ids, token_type_ids=token_type_ids, 
    position_ids=position_ids
    ) #past_key_values = lm_labels, lm_labels=lm_labels


# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(tokenizer.eos_token + input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

# text = 'Who was Jim Henson ? Jim Henson was a'

# indexed_tokens = tokenizer.encode(text)
# tokens_tensor = torch.tensor([indexed_tokens])

# model = GPT2LMHeadModel.from_pretrained(load_path)
# model.eval()

# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]

# predicted_index = torch.argmax(predictions[0, -1, :]).item()
# predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

# print(predicted_text)
