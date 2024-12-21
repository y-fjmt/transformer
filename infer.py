
import torch
import sentencepiece as spm
from dataset import WMT14_DE_EN
from train import CFG

import random

def auto_regression(
        model: torch.nn.Module,
        tokenizer: spm.SentencePieceProcessor,
        src_ids: torch.Tensor, 
    ) -> list[int]:
    """auto regression で翻訳

    Parameters
    ----------
    model : torch.nn.Module
        Transformerの翻訳モデル
    tokenizer : spm.SentencePieceProcessor
        トークナイザ
    src_ids : torch.Tensor
        入力ID列

    Returns
    -------
    list[int]
        翻訳文のID列
    """
    
    src = src_ids.unsqueeze(0)
    tgt = torch.tensor(CFG.SEQ_LEN*[tokenizer.pad_id()]).unsqueeze(0)
    
    src_msk = torch.zeros((CFG.SEQ_LEN, CFG.SEQ_LEN)).unsqueeze(0).expand(1*CFG.N_HEADS, -1, -1)
    tgt_msk = torch.nn.Transformer.generate_square_subsequent_mask(CFG.SEQ_LEN).unsqueeze(0).expand(1*CFG.N_HEADS, -1, -1)
    
    pred_ids = []
    
    for i in range(CFG.SEQ_LEN):
        
        tgt[0][i] = tokenizer.bos_id() if i == 0 else pred_ids[i-1]

        src_padding_msk = (src == tokenizer.pad_id()).float()
        tgt_padding_msk = (tgt == tokenizer.pad_id()).float()

        logits = model(src, tgt, src_mask=src_msk, 
                                tgt_mask=tgt_msk, 
                                src_padding_mask=src_padding_msk, 
                                tgt_padding_mask=tgt_padding_msk, 
                                memory_key_padding_mask=src_padding_msk)
        
        pred = torch.argmax(logits, dim=2)
        
        if pred[0][i] == tokenizer.eos_id():
            break
        else:
            pred_ids.append(pred[0][i].item())
        
    return pred_ids


if __name__ == '__main__':
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('.tokenizer/en_de_bpe_37000.model')
    
    model = torch.load('transformer-weight.pth', 
                       weights_only=False,
                       map_location=torch.device('cpu')).eval()
    
    tst_ds = WMT14_DE_EN('test', tokenizer, CFG.SEQ_LEN)
    
    idx = random.randint(0, 2999)
    src, tgt, label = tst_ds[idx]
    
    pred_ids = auto_regression(model, tokenizer, src)

    print('en:', tokenizer.decode(src.tolist()))
    print('de:', tokenizer.decode(pred_ids))