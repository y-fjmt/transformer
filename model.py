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
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerModel(nn.Module):
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model = 512):
        super(TransformerModel, self).__init__()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, batch_first=True)
        
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
    
    vocab = 10
    BS = 2
    src = torch.randint(0, 9, (BS, 16))
    tgt = torch.randint(0, 9, (BS, 16))
    label = torch.randint(0, 9, (BS, 16))
    
    # tk_en = WMT14Tokenizer('en', max_length=64, is_debug=True)
    # tk_de = WMT14Tokenizer('de', max_length=64, is_debug=True)
    # ds = WMT14_DE_EN('train', tk_en, tk_de, is_debug=True)
    # loader = DataLoader(ds)
    
    PAD_IDX = 0
    
    model = TransformerModel(vocab, vocab)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    print(src.shape)
    
    
    # (BS*num_heads, seq, seq)
    seq_len = src.shape[1]
    num_heads = 8
    src_msk = torch.zeros((seq_len, seq_len)).unsqueeze(0).expand(BS*num_heads, seq_len, seq_len)
    tgt_msk = nn.Transformer.generate_square_subsequent_mask(seq_len).unsqueeze(0).expand(BS*num_heads, -1, -1)
    
    # (N, seq)
    src_padding_msk = (src == PAD_IDX).float()
    tgt_padding_msk = (tgt == PAD_IDX).float()
    
    print('src_mask:', src_msk.shape)
    print('tgt_mask:', tgt_msk.shape)
    print('src_padding_mask:', src_padding_msk.shape)
    print('tgt_padding_mask:', tgt_padding_msk.shape)
    
    
    logits = model(src, tgt, src_mask=src_msk, tgt_mask=tgt_msk, 
                           src_padding_mask=src_padding_msk, tgt_padding_mask=tgt_padding_msk, 
                           memory_key_padding_mask=src_padding_msk)
    
    print(logits.shape)
    
    # (batch_size, seq_len, tgt_vocab) -> (batch_size * seq_len, tgt_vocab)
    logits = logits.contiguous().view(-1, vocab)
    
    # (batch_size * seq_len)
    label = label.view(-1)
    
    loss = loss_fn(logits, label)
    loss.backward()
    
    print(loss)