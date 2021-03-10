import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM,
    AdamW
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DialogueDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(model, tokenizer, opt, train_data, epochs, delay): 
    # delay: utility variable to save models after restarting colab

    losses = []

    for epoch in tqdm(range(delay, epochs + delay)):

        print('\n* Epoch {}/{}'.format(epoch+1, epochs))

        avg_loss = 0

        for batch in train_data:
            model.zero_grad()

            inputs = tokenizer(batch, return_tensors='pt', truncation=True).to(device)
            labels = inputs.input_ids.clone().detach().to(device)

            for i in range(np.random.randint(1, 5)):
                labels[0][-1 - i] = 103

            loss = model(**inputs, labels=labels).loss

            loss.backward()
            opt.step()

            avg_loss += loss / len(train_data)
        
        print('\nloss: {}'.format(avg_loss))

        losses.append(avg_loss)

        torch.save(model.state_dict(), './Models/distilbert_chat_{}'.format(epoch))
    return losses

def main():
    # fix random seeds
    np.random.seed(0)
    torch.manual_seed(0)

    dialogues = pd.read_csv('/datasets/TlkPersonaChatRus/dialogues.tsv', sep='\t')
    for column in dialogues.columns:
        dialogues[column].replace(to_replace=r'<[a-zA-Z0-9_=\/ ]+>', value=' ', regex=True, inplace=True)
    dialogues['dialogue'].replace(to_replace=r'Пользователь [12]:|Привет|Здравствуйте|[!)?,]', value='', regex=True, inplace=True)
    dialogues['dialogue'].replace(to_replace=r'\s\s+', value=' ', regex=True, inplace=True)
    data = DialogueDataset(dialogues['dialogue'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        print('Training on GPU :)')
    else:
        print('Training on CPU :(')

    tokenizer_distilbert = AutoTokenizer.from_pretrained('distilbert-base-multilingual-cased')

    model_distilbert = AutoModelForMaskedLM.from_pretrained('distilbert-base-multilingual-cased')
    model_distilbert.to(device)

    optimizer = AdamW(model_distilbert.parameters(), lr=3e-4)
    losses = train(model_distilbert, tokenizer_distilbert, optimizer, train_data, epochs=10, delay=0)
    print(losses)


if __name__ == '__main__':
    main()
