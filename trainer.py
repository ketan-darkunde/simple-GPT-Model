# Reads input.txt, trains the model, saves everything to simple_gpt_model.pt.

import torch
from model import SimpleGPT

# select GPU if avialable, else CPU for processing
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# read text and build the vocabulary
try:
    data_for_training = open('input.txt', encoding='utf-8').read()
except FileNotFoundError:
    exit("Error: file not found: input.txt")
chars  = sorted(set(data_for_training))

char_to_id = {}
for i, c in enumerate(chars):
    char_to_id[c] = i

id_to_char = {}
for i, c in enumerate(chars):
    id_to_char[i] = c

ids_list = []
for c in data_for_training:
    ids_list.append(char_to_id[c])

ids = torch.tensor(ids_list)

# model + optimizer
model = SimpleGPT(vocab_size=len(chars)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# train x = a chunk of text, y = the same chunk shifted left by one, so every position asks "what is next?"
for step in range(5000):
    starts = torch.randint(0, len(ids) - 65, (32,))

    x_list = []
    for s in starts:
        x_list.append(ids[s:s+64])
    x = torch.stack(x_list).to(device)

    y_list = []
    for s in starts:
        y_list.append(ids[s+1:s+65])
    y = torch.stack(y_list).to(device)

    # Clear old gradients
    opt.zero_grad()

    output = model(x, y)

    logits = output[0]
    loss = output[1]

    # Calculate gradients
    loss.backward()

    # Update model weights
    opt.step()

    # print("\nTraining Progress")
    # print("------------------------------")
    if step % 250 == 0:
        print(f"step {step:4d} | loss {loss.item():.4f}")
        print("------------------------------")

# save weights AND vocabulary (both are needed by chat.py)
torch.save({'weights': model.state_dict(),
            'char_to_id': char_to_id,
            'id_to_char': id_to_char},
           'simple_gpt_model.pt')

print("saved to simple_gpt_model.pt")