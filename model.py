import math
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import WMT14_DE_EN
from tokenizer import WMT14Tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model = 512):
        super(TransformerModel, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model)
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        
    
    def forward(
            self, 
            src: torch.Tensor, 
            tgt: torch.Tensor,
            src_mask: torch.Tensor,
            tgt_mask: torch.Tensor,
            src_padding_mask: torch.Tensor,
            tgt_padding_mask: torch.Tensor,
            memory_key_padding_mask: torch.Tensor
        ) -> torch.Tensor:
        
        # Embedding
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        
        # Positional Encoding
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        output = self.transformer(src_emb, tgt_emb,
                                  src_mask=src_mask,
                                  tgt_mask=tgt_mask, memory_mask=None,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        output = self.fc(output)
        return output
    
'''
正方行列の上三角部分に True、それ以外に False を持つ行列を生成する。
その後、Trueの要素を-infに、Falseの要素を0.0に置き換える。
'''
def create_mask(src, tgt, PAD_IDX):
    
    seq_len_src = src.shape[0]
    seq_len_tgt = tgt.shape[0]

    mask_tgt = generate_square_subsequent_mask(seq_len_tgt, PAD_IDX)
    mask_src = torch.zeros((seq_len_src, seq_len_src))

    padding_mask_src = (src == PAD_IDX).transpose(0, 1).float()
    padding_mask_tgt = (tgt == PAD_IDX).transpose(0, 1).float()
    
    return mask_src, mask_tgt, padding_mask_src, padding_mask_tgt


def generate_square_subsequent_mask(seq_len, PAD_IDX):
    mask = (torch.triu(torch.ones((seq_len, seq_len))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == PAD_IDX, float(0.0))
    return mask
    
    
    
if __name__ == '__main__':
    
    tk_en = WMT14Tokenizer('en', max_length=64, is_debug=True)
    tk_de = WMT14Tokenizer('de', max_length=64, is_debug=True)
    ds = WMT14_DE_EN('train', tk_en, tk_de, is_debug=True)
    loader = DataLoader(ds)
    
    PAD_IDX = 0
    
    model = TransformerModel(tk_en.vocab_size(), tk_de.vocab_size())
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    for src, tgt, label in loader:
        
        # (batch_size, seq_len) -> (seq_len, batch_size)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        print(src.shape)
        
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, PAD_IDX)
        
        # (seq_len, batch_size) -> (seq_len, batch_size, tgt_vocab)
        logits = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        

        # (seq_len, batch_size, tgt_vocab) -> (batch_size, seq_len, tgt_vocab)
        logits = logits.transpose(0, 1)
        
        # (batch_size, seq_len, tgt_vocab) -> (batch_size * seq_len, tgt_vocab)
        tgt_vocab_size = tk_de.vocab_size()
        logits = logits.contiguous().view(-1, tgt_vocab_size)
        
        # (batch_size * seq_len)
        label = label.view(-1)
        
        loss = loss_fn(logits, label)
        loss.backward()
        
        break