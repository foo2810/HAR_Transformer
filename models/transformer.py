import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn.modules.transformer import MultiheadAttention

class Transformer(nn.Module):
    def __init__(self, d_model, max_seq_len, output_dim, n_blocks=6, n_heads=1):
        super(Transformer, self).__init__()

        self.pe = PositionalEncoder(d_model, max_seq_len)

        blocks = [TransformerBlock(d_model, max_seq_len, n_heads=n_heads) for _ in range(n_blocks)]
        self.tencs = nn.Sequential(*blocks)
        # tenc = TransformerEncoderLayer(d_model, n_blocks, dim_feedforward=1024, dropout=0.1, activation='relu')
        # self.tencs = TransformerEncoder(tenc, num_layers=6)

        self.classifier = ClassificationHead(output_dim=output_dim, max_seq_len=max_seq_len, d_model=d_model)
    
    def forward(self, x, mask=None):
        x1 = self.pe(x)
        x_t = self.tencs(x1)
        x3 = self.classifier(x_t)
        return x3

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoder, self).__init__()

        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model-2, 2):
                pe[pos, i] = np.sin(pos / (10000**(2*i / d_model)))
                pe[pos, i+2] = np.cos(pos / (10000**(2*(i+1) / d_model)))
        
        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False
    
    def forward(self, x):
        ret = np.sqrt(self.d_model)*x + self.pe.to(x.device)
        return ret

class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.d_k = d_model
    
    def forward(self, q, k, v, mask=None):
        k = self.k_linear(k)
        q = self.k_linear(q)
        v = self.k_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / np.sqrt(self.d_k)

        mask = mask.unsqueeze(1)
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        normalized_weights = F.softmax(weights, dim=-1)

        output = torch.matmul(normalized_weights, v)
        output = self.out(output)

        return output, normalized_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, max_seq_len, d_ff=1024, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, max_seq_len, n_heads=1, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        #self.attn = Attention(d_model)
        self.attn = MultiheadAttention(d_model, num_heads=n_heads, dropout=0.)

        self.ff = FeedForward(d_model, max_seq_len)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x1, normalized_weights = self.attn(x, x, x, mask)
        # x1 = self.norm_1(x + self.dropout_1(x1))
        # x2 = self.ff(x1)
        # x2 = self.norm_2(x1 + self.dropout_2(x2))
        # output = x2

        x_normalized = self.norm_1(x)
        output, normalized_weights = self.attn(x_normalized, x_normalized, x_normalized)#, mask)

        x2 = x + self.dropout_1(output)

        x_normalized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normalized2))

        return output#, normalized_weights

class ClassificationHead(nn.Module):
    def __init__(self, d_model, max_seq_len, output_dim):
        super(ClassificationHead, self).__init__()

        self.linear = nn.Linear(max_seq_len*d_model, output_dim)
        #self.linear = nn.Linear(d_model, output_dim)

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, std=0.1)
    
    def forward(self, x):
        # x0 = x[:, 0, :]
        # x0 = torch.sum(x, dim=1)
        x0 = x.view(x.size(0), -1)
        out = self.linear(x0)
        return out
