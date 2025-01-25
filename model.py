import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    
    def __init__(
            self, 
            vocab_size, 
            seq_len = 128,
            d_model = 512,
            nhead = 8,
            num_encoder_layers = 6,
            num_decoder_layers = 6,
            dim_feedforward = 2048,
            dropout = 0.1,
        ) -> None:
        super(TransformerModel, self).__init__()
        
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=seq_len)
        self.emb = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True)
        
        self.fc = nn.Linear(d_model, vocab_size)
        
        
    def enc_forward(
            self,
            src_ids,
            src_mask, 
            src_key_padding_mask, 
            src_is_causal = None
    ) -> torch.Tensor:
        
        src = self.emb(src_ids)
        src = self.pos_enc(src)
        
        return self.transformer.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )
        
    
    def dec_forward(
        self,
        tgt_ids, memory,
        tgt_mask, memory_mask,
        tgt_key_padding_mask, memory_key_padding_mask,
        tgt_is_causal = None, memory_is_causal=False,
    ) -> torch.Tensor:
        
        tgt = self.emb(tgt_ids)
        tgt = self.pos_enc(tgt)
        
        return self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        
    
    def forward(
            self, 
            src_ids: torch.Tensor, 
            tgt_ids: torch.Tensor,
            src_mask: torch.Tensor,
            tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor,
            tgt_padding_mask: torch.Tensor,
            memory_key_padding_mask: torch.Tensor
        ) -> torch.Tensor:
        
        memory = self.enc_forward(src_ids,
                                  src_mask, 
                                  src_padding_mask)
        
        dec_out = self.dec_forward(tgt_ids, memory,
                                   tgt_mask, src_mask,
                                   tgt_padding_mask, src_padding_mask)
        
        output = self.fc(dec_out)
        return output
