import torch
import torch.nn as nn
import numpy as np

class PosEmbedding(nn.Module):
    def __init__(self, d_model, seq_length, number_tokens, n=10000):
        super(PosEmbedding, self).__init__()
        self.n = n
        self.d_model = d_model
        self.number_tokens = number_tokens
        self.seq_length = seq_length

        # Create embedding layer
        self.embedding = nn.Embedding(number_tokens, d_model)

        # Generate positional encoding
        pos_encoding = self.get_pos_encodings(seq_length=seq_length)

        # Register as buffer
        self.register_buffer("pos_encoding", pos_encoding)

        # Make non-trainable
        self.pos_encoding.requires_grad = False
    
    def get_pos_encodings(self, seq_length):
        encoding = torch.zeros(seq_length, self.d_model, dtype=torch.float32)
        position = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-np.log(self.n) / self.d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        # (1, seq_length, d_model)
        return encoding.unsqueeze(0)  

    def forward(self, input):
        # input: (batch_size, seq_length)

        #embedding: (batch_size, seq_length, d_model)
        embedding = self.embedding(input)  

        # Slice pos_encoding to match input sequence length
        seq_len = input.size(1)
        pos_encoding = self.pos_encoding[:, :seq_len, :]  

        # Expand to match batch size
        pos_encoding = pos_encoding.expand(input.size(0), seq_len, self.d_model)

        # Add positional encodings to embeddings
        output = embedding + pos_encoding
        return output