import torch
from torch import nn
import torch.nn.functional as F

from dataset import WMT14_DE_EN
from tokenizer import WMT14Tokenizer

class TransformerModel(nn.Module):
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model = 512):
        super(TransformerModel, self).__init__()
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model)
        self.fc1 = nn.Linear(d_model, 2048)
        self.fc2 = nn.Linear(2048, d_model)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        
        # (n_word, 1) -> (n_word, 512)
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        
        # (n_word_src, 512), (n_word_tgt, 512) -> (n_word_tgt, 512)
        x = self.transformer(src=src_emb, tgt=tgt_emb)
        
        # (n_word_tgt, 512) -> (n_word_tgt, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x
    
    
    
if __name__ == '__main__':
    pass