import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoid positional embedding
    Args:
        d_model (int): embedding dimension
        dropout (float): dropout rate of final computation
        dropout_emb (float): dropout rate of the positional embedding
        max_len (int): maximum input length
        scale (bool): where to scale the input by sqrt(d_model) or not
    """
    def __init__(self, d_model, dropout=0.1, dropout_emb=0.0, max_len=100, scale=True):
        """Construct a Sinusoid positional embedding follow by original transformer"""
        super().__init__()
        self._dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.d_model = d_model
        self.dropout_emb = dropout_emb
        self.scale = scale

        if self.dropout_emb > 0:
            self._dropout_emb = nn.Dropout(self.dropout_emb)
        else:
            self._dropout_emb = None

        self.set_pe()

    def set_pe(self):
        """if num_patches < self.max_len:
            self.max_len """
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                             (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale:
            x = x * math.sqrt(self.d_model)
        # print("input to positional embedding: ", x.shape)
        # print(self.pe.shape)
        pos_emb = self.pe[:x.size(0), :]
        # print("pos_emb :",pos_emb.shape)
        if self.dropout_emb:
            pos_emb = self._dropout_emb(pos_emb)
        x = x + pos_emb
        x = self._dropout(x)
        return x
    
if __name__ == "__main__":
    d_model = 768
    pe = PositionalEncoding(
        d_model = d_model,
        max_len = 70,
    )
    tok_emb = torch.randn(70, 1, d_model)
    print(pe(tok_emb).shape)
