import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, h, dk, dv, d_model):
        super(MultiHeadAttention, self).__init__()
        self.h = h 
        self.dk = dk 
        self.dv = dv 
        self.d_model = d_model

        # Linear layers for query, key, value projections
        self.wq = nn.Linear(d_model, h * dk)
        self.wk = nn.Linear(d_model, h * dk)
        self.wv = nn.Linear(d_model, h * dv)

        # Output linear layer
        self.wo = nn.Linear(h * dv, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_length = query.size(1)

        Q = self.wq(query)  # (batch_size, seq_length, h * dk)
        K = self.wk(key)    # (batch_size, seq_length, h * dk)
        V = self.wv(value)  # (batch_size, seq_length, h * dv)

        # Reshape: (batch_size, seq_length, h, dk/dv) -> (batch_size, h, seq_length, dk/dv)
        Q = Q.view(batch_size, seq_length, self.h, self.dk).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.h, self.dk).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.h, self.dv).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dk ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)

         # (batch_size, h, seq_length, dv)
        values = torch.matmul(attention, V) 

        # (batch_size, h, seq_length, dv) -> (batch_size, seq_length, h * dv)
        values = values.transpose(1, 2).contiguous().view(batch_size, seq_length, self.h * self.dv)

        # (batch_size, seq_length, d_model)
        output = self.wo(values)  
        return output