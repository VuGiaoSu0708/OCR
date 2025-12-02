import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2):  # Tăng dropout
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Thêm layer scale để cải thiện học sâu
        self.gamma1 = nn.Parameter(torch.ones(d_model) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(d_model) * 0.1)
        self.gamma3 = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, return_attn=False):
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=True)
        tgt = tgt + self.dropout1(self.gamma1 * tgt2)  # Add layer scale
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask, need_weights=True)
        tgt = tgt + self.dropout2(self.gamma2 * tgt2)  # Add layer scale
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))  # GELU thay vì ReLU
        tgt = tgt + self.dropout3(self.gamma3 * tgt2)  # Add layer scale
        tgt = self.norm3(tgt)
        if return_attn:
            return tgt, self_attn_weights, cross_attn_weights
        return tgt
