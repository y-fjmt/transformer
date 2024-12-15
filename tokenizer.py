from typing import Literal
from datasets import load_dataset
from tqdm import tqdm
import re

from dataset import WMT14_DE_EN

class WMT14Tokenizer:
    
    special_tokens = [
        '<SOS>',
        '<EOS>',
        '<PAD>',
        '<UKN>'
    ]
    
    
    def __init__(
            self, 
            lang: Literal['en', 'de'], 
            max_length: int = 256,
            known_cnt: int = 500,
            is_debug: bool = False
        ) -> None:
        """tokenizerのインスタンスを作成
        """
        
        self.max_length = max_length
        vocab_count = {}
        
        dataset = load_dataset('wmt14', 'de-en', split='train')
        
        if is_debug:
            known_cnt = 10
            dataset = dataset.sort('translation')
            dataset = dataset.select(range(1000))
        
        for col in tqdm(dataset):
            text: str = col['translation'][lang]
            
            tokens = self.tokenize(text, with_special_token=False)
            for token in tokens:
                if token not in vocab_count:
                    vocab_count[token] = 1
                else:
                    vocab_count[token] += 1
                    
            
        common_words = sorted([word for word, cnt in vocab_count.items() if cnt >= known_cnt])
        self.id_to_word = self.special_tokens + common_words
        self.word_to_id = {word: idx for idx, word in enumerate(self.id_to_word)}
    
    
    def tokenize(
            self, 
            text: str, 
            with_special_token: bool = True
        ) -> list[str]:
        """文字列をトークン(単語)に分割
        """
        
        text = self._preprocess(text)
        words = text.split()
        
        # max_length を超えた場合はクリップ
        if len(words) > self.max_length:
            words = words[:self.max_length]
        
        # add special tokens
        if with_special_token:
            
            for i in range(len(words)):
                if words[i] not in self.id_to_word:
                    words[i] = '<UKN>'
                
            pads = ['<PAD>'] * (self.max_length - len(words))
            words += pads
            words = ['<SOS>'] + words[:-2] + ['<EOS>']
            
        return words
    
    
    def encode(self, text: str) -> list[int]:
        """トークン化してid列に変換
        """
        tokens = self.tokenize(text)
        return [self.word_to_id[word] for word in tokens]
    
    
    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """id列をトークン列に戻す
        """
        return [self.id_to_word[idx] for idx in ids]
    
    
    def _preprocess(self, text: str) -> str:
        """すべて小文字にしてアルファベットのみにする
        """
        text = text.lower()
        text = re.sub('[^a-z]', ' ', text)
        return text
    
    
    def vocab_size(self) -> int:
        
        return len(self.id_to_word)
    
    
if __name__ == '__main__':
    pass
    
    