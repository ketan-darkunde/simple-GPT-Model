# Loads the saved model and continues whatever you type.

import torch
import torch.nn.functional as F
from model import SimpleGPT

# select GPU if avialable, else CPU for processing
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

saved = torch.load('simple_gpt_model.pt', map_location=device)
model = SimpleGPT(vocab_size=len(saved['char_to_id'])).to(device)
model.load_state_dict(saved['weights'])
model.eval()
char_to_id, id_to_char = saved['char_to_id'], saved['id_to_char']

@torch.no_grad()

def generate(prompt, n_chars=30):
    if any(c not in char_to_id for c in prompt):
        return "(you used characters that weren't in the training text)"
    
    out = [char_to_id[c] for c in prompt]

    for _ in range(n_chars):
        x = torch.tensor([out[-model.block_size:]], device=device)
        logits, _ = model(x)
        probs = F.softmax(logits[0, -1], dim=-1)
        out.append(torch.multinomial(probs, 1).item()) # sample one character
        
    return ''.join(id_to_char[i] for i in out)

print('To exit from chat, enter quit() or exit() or bye()')

while True:
    q = input("You: ")
    if q.strip().lower() in ('quit()', 'exit()', 'bye()'):
        break
    elif(q == "" or q == " "):
        print('ask question')
    else:
        print("SimpleGPT:", generate(q))