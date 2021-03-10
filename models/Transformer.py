import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, model_size, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads
        self.linears = nn.ModuleList([
            nn.Linear(model_size, model_size) for _ in range(4)
        ])
        self.attention_score = None
        
 
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
 
        n_batches = query.size(0) 
        query, key, value = [l(x).view(nbatches, -1, self.n_heads, self.emb_size).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
 
        logits = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(emb_size)
 
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
 
        self.attention_score = F.softmax(logits, dim=-1)
        x = torch.matmul(self.attention_score, value)
        x = x.transpose(1, 2).contiguous().view(
            n_batches, -1, self.n_heads * self.emb_size)
        
        return self.linears[-1](x)

class Encoder(nn.Module): 
    def __init__(self, n_attention_heads):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(n_attention_heads)])
        self.norm = LayerNorm(layer.size)
 
 
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class AddPlusNormLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
 
    def forward(self, x, sublayer):
        norm = self.a_2 * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6) + self.b_2
        return x + norm

class FeedForwardLayer(nn.Module):
    def __init__(self, size, self_attention, feed_forward):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([
            copy.deepcopy(SublayerConnection(size)) for _ in range(2)])
        self.size = size
 
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class PositionalEncoding(nn.Module):
    def __init__(self, model_size, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_size, 2) * (-(torch.log(10000.0) / model_size)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            layer for _ in range(N)
        ])
        self.norm = AddPlusLayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

