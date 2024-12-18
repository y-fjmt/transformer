from typing import Literal
from datasets import load_dataset
from tqdm import tqdm
import os
import re
import pickle


class WMT14Tokenizer:
    
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    unk_token = '<unk>'
    
    special_tokens = [
        pad_token,
        sos_token,
        eos_token,
        unk_token
    ]
    
    dir_name = '.wmt14tokenizer'
    
    
    def __init__(
            self, 
            lang: Literal['en', 'de'], 
            max_length: int = 128,
            known_cnt: int = 500,
            is_debug: bool = False,
            hide_pbar: bool = False
        ) -> None:
        """tokenizerのインスタンスを作成
        """
        
        self.max_length = max_length
        vocab_count = {}
        
        dataset = load_dataset('wmt14', 'de-en', split='train')
        
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
            
        path = f'{self.dir_name}/{lang}.pickle'
        if os.path.isfile(path):
            # 学習済みファイルがあれば読み込む
            with open(path, 'rb') as fp:
                clf = pickle.load(fp)
                self.id_to_word = clf['id_to_word']
                self.word_to_id = clf['word_to_id']
        else:
            # 学習済みファイルがなければ作成
            for col in tqdm(dataset, disable=hide_pbar):
                text: str = col['translation'][lang]
                
                tokens = self.tokenize(text, with_pad=False, 
                                            with_sos=False, 
                                            with_eos=False, 
                                            with_ukn=False,
                                            ignore_maxlen=True)
                for token in tokens:
                    if token not in vocab_count:
                        vocab_count[token] = 1
                    else:
                        vocab_count[token] += 1
                        
                
            common_words = sorted([word for word, cnt in vocab_count.items() if cnt >= known_cnt])
            self.id_to_word = self.special_tokens + common_words
            self.word_to_id = {word: idx for idx, word in enumerate(self.id_to_word)}
            
            with open(path, 'wb') as fp:
                clf = {
                    'id_to_word': self.id_to_word,
                    'word_to_id': self.word_to_id
                }
                pickle.dump(clf, fp)
    
    
    def tokenize(
            self, 
            text: str, 
            with_pad: bool = True,
            with_sos: bool = True,
            with_eos: bool = True,
            with_ukn: bool = True,
            ignore_maxlen: bool = False
        ) -> list[str]:
        """文字列をトークン(単語)に分割
        """
        
        text = self._preprocess(text)
        words = text.split()
        
        # max_length を超えた場合はクリップ
        if not ignore_maxlen:
            n_sos_eos = int(with_sos) + int(with_eos)
            if len(words) > self.max_length - n_sos_eos:
                words = words[:self.max_length - n_sos_eos]
        
        # add <ukn>
        if with_ukn:
            for i in range(len(words)):
                if words[i] not in self.id_to_word:
                    words[i] = self.unk_token
        
        # add <sos>
        if with_sos:
            words = [self.sos_token] + words
            
        # add <eos>
        if with_eos:
            words = words + [self.eos_token]
        
        # add <pad>
        if with_pad:
            pads = [self.pad_token] * (self.max_length - len(words))
            words += pads
            
        return words
    
    
    def encode(
            self, 
            text: str,
            with_pad: bool = True,
            with_sos: bool = True,
            with_eos: bool = True,
            with_ukn: bool = True
        ) -> list[int]:
        """トークン化してid列に変換
        """
        tokens = self.tokenize(text, with_pad, with_sos, with_eos, with_ukn)
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
        """語彙数を取得
        """
        return len(self.id_to_word)
    
    
if __name__ == '__main__':
    pass
    