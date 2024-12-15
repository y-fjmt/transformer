import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from tokenizer import WMT14Tokenizer


class WMT14_DE_EN(Dataset):
    
    split_map = {
        'train': 'train',
        'val': 'validation',
        'test': 'test'
    }
    
    def __init__(
            self, 
            split: Literal['train', 'val', 'test'], 
            src_tokenizer: 'WMT14Tokenizer',
            tgt_tokenizer: 'WMT14Tokenizer',
            is_debug: bool = False
        ) -> None:
        
        split_name = self.split_map[split]
        self.hf_dataset = load_dataset("wmt14", "de-en", split=split_name)
        
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        if is_debug:
            self.hf_dataset = self.hf_dataset.sort('translation')
            self.hf_dataset = self.hf_dataset.select(range(1000))
    
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        translation = self.hf_dataset[idx]['translation']
        src, tgt = translation['en'], translation['de']
        
        src = self.src_tokenizer.encode(src)
        tgt = self.tgt_tokenizer.encode(tgt)
        
        return torch.tensor(src), torch.tensor(tgt)

        


if __name__ == '__main__':
    pass