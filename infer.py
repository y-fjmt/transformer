import torch
from model import TransformerModel
from dataset import WMT14_DE_EN
from tokenizer import WMT14Tokenizer
from train import CFG

PATH = 'transformer-weight.pth'

if __name__ == '__main__':
    
    src_tokenizer = WMT14Tokenizer(lang='en', max_length=CFG.SEQ_LEN)
    tgt_tokenizer = WMT14Tokenizer(lang='de', max_length=CFG.SEQ_LEN)

    test_ds = WMT14_DE_EN('test', src_tokenizer, tgt_tokenizer)
    
    
    src_vocab = src_tokenizer.vocab_size()
    tgt_vocab = tgt_tokenizer.vocab_size()

    model = TransformerModel(src_vocab, tgt_vocab)
    model.load_state_dict(torch.load(CFG.SAVE_AS, weights_only=False), )
    model.eval()