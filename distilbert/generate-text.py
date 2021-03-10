import torch
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from tqdm import tqdm


model = AutoModelForMaskedLM.from_pretrained("distilbert-base-multilingual-cased")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.load_state_dict(torch.load('distilbert_chat', map_location=device))

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

sentence = ' '.join(sys.argv[1:])
r = np.random.randint(low=2, high=5)
for _ in range(r):
	t = fill_mask(sentence + '[MASK]')
	sentence += ' ' + t[0]['token_str']

print(sentence)

