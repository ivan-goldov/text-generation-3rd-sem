import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',   
]
 
data = ' '.join(corpus)

class TextPreprocessor:
    def __init__(self):
        self.vocabulary = None
        self.tokens = None
 
    def tokenizer(self, data):
        return data.split()
 
    def make_vocabulary(self, data):
        tmp = set()
        self.tokens = self.tokenizer(data)
        for i in self.tokens:
            tmp.add(i)
        self.vocabulary = list(tmp)

class Word2Vec(nn.Module):
    def __init__(self, vocabulary, tokens, window_size=2, emb_size=30):
        super(Word2Vec, self).__init__()
        self.window_size = window_size
        self.emb_size = emb_size
        self.tokens = tokens
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        # self.w1 = torch.rand((self.vocabulary_size, self.emb_size))
        # self.w2 = torch.rand((self.emb_size, self.vocabulary_size))
        self.fc1 = nn.Linear(self.vocabulary_size, self.emb_size)
        self.fc2 = nn.Linear(self.emb_size, self.vocabulary_size)
        self.word2id = {w: id for (id, w) in enumerate(self.vocabulary)}
        self.id2word = {id: w for (id, w) in enumerate(self.vocabulary)}
        self.id_pairs = None
 
 
    def word_onehot(self, word):
        vector = np.zeros((self.vocabulary_size))
        vector[self.word2id[word]] = 1
        return torch.from_numpy(vector).float()
    
 
    def compute_id_pairs(self):
        self.id_pairs = []
        for center_word_id in range(2, len(self.tokens) - 2):
            center_vector = self.word_onehot(self.tokens[center_word_id])
            for i in range(-self.window_size, self.window_size+1):
                if i != 0:
                    context_word_id = center_word_id + i
                    context_vector = self.word_onehot(self.tokens[context_word_id])
                    self.id_pairs.append((center_vector, context_vector))
 
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
 
 
    def word_2_vec(self, word):
        onehot = self.word_onehot(word)
        return self.fc1(onehot).detach().numpy()
 
 
    def word_sim(self, word1, word2):
        oh1 = self.word_onehot(word1)
        oh2 = self.word_onehot(word2)
        v1 = self.fc1(oh1).detach().numpy()
        v2 = self.fc1(oh2).detach().numpy()
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))