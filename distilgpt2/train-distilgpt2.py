import torch
import pandas as pd
import numpy as np


from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, 
                          AutoModelWithLMHead, 
                          Trainer, 
                          TrainingArguments, 
                          DataCollatorForLanguageModeling,
                          LineByLineTextDataset
                          )


class DialogueDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train():
    dialogues = pd.read_csv('/content/TlkPersonaChatRus/dialogues.tsv', sep='\t')
    for column in dialogues.columns:
        dialogues[column].replace(to_replace=r'<[a-zA-Z0-9_=\/ ]+>', value=' ', regex=True, inplace=True)
    dialogues['dialogue'].replace(to_replace=r'Пользователь [12]:|Привет|Здравствуйте|[!)?,]', value='', regex=True, inplace=True)
    dialogues['dialogue'].replace(to_replace=r'\s\s+', value=' ', regex=True, inplace=True)
    dialogues = dialogues['dialogue']
    dialogues.to_csv('./Datasets/dialogues')

    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    model = AutoModelWithLMHead.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token

    dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='/content/drive/MyDrive/semester-practice-3rd/Datasets/dialogues.txt',
    block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='/content/drive/MyDrive/semester-practice-3rd/Models/distilgpt2',
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model('model/gpt2_chat')


if __name__ == '__main__':
    train()
