import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from ..layers.positional_encoding import PositionalEncoding
from ..layers.transformer_decoder import TransformerDecoderLayer
from ..config import CHAR2IDX, IDX2CHAR


class TextRecognition(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_decoder_layers=3, max_length=30):
        super().__init__()
        # Encoder
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool & fc
        self.enc_proj = nn.Linear(2048, d_model)
        self.enc_pos = PositionalEncoding(d_model, max_len=64*8)  # 64x8=512 patches

        # Decoder
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dec_pos = PositionalEncoding(d_model, max_len=max_length)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead) for _ in range(num_decoder_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_length = max_length

    def forward(self, images, tgt_seq, tgt_mask=None, return_attn=False):
        # CNN Encoder
        features = self.cnn(images)  # [B, 512, 2, 8]
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, 16, 512]
        features = self.enc_proj(features)  # [B, 16, d_model]
        features = self.enc_pos(features)

        # Decoder
        tgt_emb = self.embedding(tgt_seq)
        tgt_emb = self.dec_pos(tgt_emb)
        x = tgt_emb
        attn_maps = []
        for layer in self.decoder_layers:
            if return_attn:
                x, self_attn, cross_attn = layer(x, features, tgt_mask=tgt_mask, return_attn=True)
                attn_maps.append(cross_attn)
            else:
                x = layer(x, features, tgt_mask=tgt_mask)
        out = self.fc_out(x)
        if return_attn:
            return out, attn_maps
        return out

    def inference(self, image, sos_idx=None, eos_idx=None, return_attn=False):
        if sos_idx is None:
            sos_idx = CHAR2IDX['<SOS>']
        if eos_idx is None:
            eos_idx = CHAR2IDX['<EOS>']
            
        self.eval()
        with torch.no_grad():
            # Encode
            features = self.cnn(image.unsqueeze(0))
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)
            features = self.enc_proj(features)
            features = self.enc_pos(features)
            
            # Decode with temperature sampling
            tgt_seq = torch.tensor([[sos_idx]], device=image.device)
            outputs = []
            attn_maps = []
            for _ in range(self.max_length):
                tgt_emb = self.embedding(tgt_seq)
                tgt_emb = self.dec_pos(tgt_emb)
                x = tgt_emb
                for layer in self.decoder_layers:
                    if return_attn:
                        x, self_attn, cross_attn = layer(x, features, return_attn=True)
                        attn_maps.append(cross_attn)
                    else:
                        x = layer(x, features)
                out = self.fc_out(x)
                
                # Temperature sampling
                logits = out[:, -1, :] / 0.7  # temperature = 0.7
                probs = F.softmax(logits, dim=-1)
                
                # Top-k sampling (k=5)
                topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
                topk_probs = topk_probs / topk_probs.sum()  # Re-normalize
                next_token = topk_indices[0, torch.multinomial(topk_probs[0], 1)]
                
                outputs.append(next_token.unsqueeze(0))
                # Fix: Ensure next_token is properly shaped before concatenation
                next_token = next_token.view(1, 1)  # Reshape to [1, 1]
                tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
                if next_token.item() == eos_idx:
                    break
            
            output_seq = torch.cat(outputs, dim=0)
            pred_text = ''.join([IDX2CHAR[idx.item()] for idx in output_seq 
                                if idx.item() not in [CHAR2IDX['<PAD>'], CHAR2IDX['<SOS>'], CHAR2IDX['<EOS>']]])
            if return_attn:
                return output_seq, pred_text, attn_maps
            return output_seq, pred_text
