
import os
from datasets import load_dataset

import sentencepiece as spm
from tqdm import tqdm
from typing import Literal

def train(
        lang: Literal['en', 'de', 'en_de'] = 'en_de', 
        vocab_size: int = 37000, 
        algolithm: Literal['unigram', 'bpe'] = 'bpe'
    ) -> None:
    
    trn_ds = load_dataset('wmt14', 'de-en', split='train')
    dir_name = '.tokenizer'
    file_name = f'{dir_name}/{lang}_{algolithm}_{vocab_size}'
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    with open(f'{file_name}.txt', 'w') as fp:
        for col in tqdm(trn_ds):
            if lang == 'en_de':
                fp.write(col['translation']['en']+ '\n')
                fp.write(col['translation']['de']+ '\n')
            else:
                fp.write(col['translation'][lang]+ '\n')
      
    spm.SentencePieceTrainer.Train(
        input=f'{file_name}.txt',
        model_prefix=file_name,
        vocab_size=vocab_size,
        model_type=algolithm,
        pad_id=3,
        pad_piece='<pad>',
        )
    
if __name__ == '__main__':
    train()