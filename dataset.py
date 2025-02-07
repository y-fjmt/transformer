import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
import sentencepiece as spm

from torch.utils.data import DataLoader

from typing import Literal
from config import CFG


class WMT14_DE_EN(Dataset):
    
    split_map = {
        'train': 'train',
        'val': 'validation',
        'test': 'test'
    }
    
    def _add_langth(self, col):
        
        translation = col['translation']
        src, tgt = translation['de'], translation['en']
        
        n_src_ids = len(self.tokenizer.EncodeAsIds(src))
        n_tgt_ids = len(self.tokenizer.EncodeAsIds(tgt))
        length = max(n_src_ids, n_tgt_ids)
        
        return {
                'translation': {'de': src, 'en': tgt},
                'length': length
            }
    
    def __init__(
            self, 
            split: Literal['train', 'val', 'test'], 
            tokenizer: spm.SentencePieceProcessor,
            is_debug: bool = False
        ) -> None:
        
        split_name = self.split_map[split]
        self.tokenizer = tokenizer
        self.hf_dataset = load_dataset("wmt14", "de-en", split=split_name)
        
        if is_debug:
            n_retrieve = (10000 if split == 'train' else 3000)
            self.hf_dataset = self.hf_dataset.sort('translation')
            self.hf_dataset = self.hf_dataset.select(range(n_retrieve))
            
        # バッチの長さを揃えやすくするためにソート
        self.hf_dataset = self.hf_dataset.map(self._add_langth, batched=False, num_proc=CFG.N_WORKERS)
        self.hf_dataset = self.hf_dataset.sort('length', reverse=True)
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        translation = self.hf_dataset[idx]['translation']
        src, tgt = translation['de'], translation['en']
        
        # sentence -> ids with special tokens
        src_ids = self.tokenizer.EncodeAsIds(src, add_bos=True, add_eos=True)
        tgt_ids = self.tokenizer.EncodeAsIds(tgt, add_bos=True, add_eos=True)
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)
    
    
class Collator:
    
    def __init__(self, pad_idx, max_len=CFG.SEQ_LEN):
        self.pad_idx = pad_idx

    def __call__(
            self,
            batch: list[torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """長さの不揃いなseqをpadしてバッチにする
            tgtをdeconderの入力と出力に分割

        Args:
            batch (list[torch.Tensor]): datasetの出力

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        
        src, tgt, label = [], [], []
        
        for src_ids, tgt_ids in batch:
            src.append(src_ids)
            tgt.append(tgt_ids[:-1])
            label.append(tgt_ids[1:])
        
        padded = pad_sequence(
                        # まとめると src, tgt の長さが揃う
                        src+tgt+label, 
                        batch_first=True, padding_value=self.pad_idx)
        
        # clop
        _, S = padded.shape
        if S > CFG.SEQ_LEN:
            padded = padded[:, :CFG.SEQ_LEN]
        
        src, tgt, label = padded.chunk(3)
            
        return src, tgt, label




if __name__ == '__main__':
    pass