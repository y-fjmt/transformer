import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import sentencepiece as spm

from typing import Literal


class WMT14_DE_EN(Dataset):
    
    split_map = {
        'train': 'train',
        'val': 'validation',
        'test': 'test'
    }
    
    def __init__(
            self, 
            split: Literal['train', 'val', 'test'], 
            tokenizer: spm.SentencePieceProcessor,
            max_seq_len = 128,
            is_debug: bool = False
        ) -> None:
        
        split_name = self.split_map[split]
        self.hf_dataset = load_dataset("wmt14", "de-en", split=split_name)
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if is_debug:
            n_retrieve = (10000 if split == 'train' else 3000)
            self.hf_dataset = self.hf_dataset.sort('translation')
            self.hf_dataset = self.hf_dataset.select(range(n_retrieve))
    
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        translation = self.hf_dataset[idx]['translation']
        src, tgt = translation['en'], translation['de']
        
        # src_ids = self.src_tokenizer.encode(src)
        # tgt_ids = self.tgt_tokenizer.encode(tgt, with_eos=False)
        # label_ids = self.tgt_tokenizer.encode(tgt, with_sos=False)
        src_ids = self.tokenizer.Encode(src, out_type=int, add_bos=True, add_eos=True)
        tgt_ids = self.tokenizer.Encode(tgt, out_type=int, add_bos=True)
        label_ids = self.tokenizer.Encode(tgt, out_type=int, add_eos=True)
        
        if len(src_ids) > self.max_seq_len:
            src_ids = src_ids[:self.max_seq_len]
        
        if len(tgt_ids) > self.max_seq_len:
            tgt_ids = tgt_ids[:self.max_seq_len]
            
        if len(label_ids) > self.max_seq_len:
            label_ids = label_ids[:self.max_seq_len]
            
        # padding
        pad_id = self.tokenizer.pad_id()
        src_ids +=  [pad_id] * (self.max_seq_len - len(src_ids))
        tgt_ids +=  [pad_id] * (self.max_seq_len - len(tgt_ids))
        label_ids +=  [pad_id] * (self.max_seq_len - len(label_ids))
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids), torch.tensor(label_ids)

        


if __name__ == '__main__':
    pass