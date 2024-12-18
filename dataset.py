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
            n_retrieve = (10000 if split == 'train' else 3000)
            self.hf_dataset = self.hf_dataset.sort('translation')
            self.hf_dataset = self.hf_dataset.select(range(n_retrieve))
    
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        translation = self.hf_dataset[idx]['translation']
        src, tgt = translation['en'], translation['de']
        
        src_ids = self.src_tokenizer.encode(src)
        tgt_ids = self.tgt_tokenizer.encode(tgt, with_eos=False)
        label_ids = self.tgt_tokenizer.encode(tgt, with_sos=False)
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids), torch.tensor(label_ids)

        


if __name__ == '__main__':
    pass