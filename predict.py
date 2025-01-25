import torch
from torch import nn
from tqdm import tqdm
import sentencepiece as spm
from torch.utils.data import DataLoader
from model import TransformerModel
from dataset import WMT14_DE_EN

from torcheval.metrics import BLEUScore

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def beam_search(
        model: TransformerModel,
        tokenizer: spm.SentencePieceProcessor,
        src_ids: torch.Tensor, 
        seq_len: int = 128,
        n_beams: int = 4
    ) -> torch.Tensor:
    """ビームサーチで翻訳

    Args:
        model (TransformerModel): 学習済みのモデル
        tokenizer (spm.SentencePieceProcessor): トークナイザ
        src_ids (torch.Tensor): 入力の単語列 (S, ) or (N, S)
        seq_len (int, optional): シーケンス長. Defaults to 128.
        n_beams (int, optional): ビーム数. Defaults to 4.

    Returns:
        torch.Tensor: 対数尤度の最も高い結果 (S, ) or (N, S)
    """
    
    assert src_ids.device == next(model.parameters()).device, \
        '`src_ids` and `model` must be same device.'
    
    assert src_ids.dim() == 1 or src_ids.dim() == 2, \
        f'`src_ids` must be 1D or 2D tensor, got {src_ids.dim()}D tensor.'
    
    device = src_ids.device
    is_batched = (src_ids.dim() == 2)
    if not is_batched:
        src_ids = src_ids.unsqueeze(0)
        
    batch_size, _ = src_ids.shape
    
    src_msk = torch.zeros(seq_len, seq_len, device=device)
    src_padding_msk = (src_ids == tokenizer.pad_id()).float()
    
    tgt_ids = torch.full_like(src_ids, tokenizer.pad_id(), device=device)
    tgt_ids[:, 0] = tokenizer.bos_id()
    tgt_msk = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
    # エンコーダの計算
    memory = model.enc_forward(src_ids, src_msk, src_padding_msk)
    
    tgt_log_likehood = torch.full((batch_size, n_beams), 0.0, device=device)
    has_eos = torch.full((batch_size, n_beams), False, device=device)
    
    # デコーダを再帰的に計算
    for i in range(seq_len-1):
        
        if i == 0:
            
            tgt_padding_msk = (tgt_ids == tokenizer.pad_id()).float()

            logit = model.fc(
                model.dec_forward(tgt_ids, memory, 
                                tgt_msk, src_msk, 
                                tgt_padding_msk, src_padding_msk)
            )
            outputs = nn.functional.softmax(logit, dim=2)
            
            # 上位 n_beams 個の token と 確率をとる
            # (N, n_beams)
            top_value, top_idx = torch.topk(outputs, n_beams)
            proba = top_value[:, i, :]
            candidates = top_idx[:, i, :]
            
            # (N, S) -> (N, n_beams, S)
            tgt_ids = tgt_ids.unsqueeze(1).repeat(1, n_beams, 1)
            tgt_ids[:, :, i+1] = candidates
            tgt_log_likehood += torch.log(proba)
            
            # 次のイテレーションに備える
            # (N, S, emb) -> (N*n_beams, S, emb)
            memory = memory.unsqueeze(1).repeat(1, n_beams, 1, 1).view(batch_size*n_beams, seq_len, -1)
            src_padding_msk = src_padding_msk.unsqueeze(1).repeat(1, n_beams, 1).view(batch_size*n_beams, seq_len)
            
        else:
            
            # (N, S)の形にする
            tgt_ids = tgt_ids.view(-1, seq_len)
            
            tgt_padding_msk = (tgt_ids == tokenizer.pad_id()).float()

            # 次の単語を予測
            logit = model.fc(
                model.dec_forward(tgt_ids, memory, 
                                tgt_msk, src_msk, 
                                tgt_padding_msk, src_padding_msk)
            )
            outputs = nn.functional.softmax(logit, dim=2)
            
            # 上位 n_beams 個の token と 確率をとる
            # (N*n_beams, n_beams)
            top_value, top_idx = torch.topk(outputs, n_beams)
            proba = top_value[:, i, :]
            candidates = top_idx[:, i, :]
            
            # (N*n_beams, n_beams) -> (N, n_beams, n_beams)
            candidates = candidates.view(batch_size, n_beams, n_beams)
            proba = proba.view(batch_size, n_beams, n_beams)
            
            # (N*n_beams, S) -> (N, n_beams, S)
            tgt_ids = tgt_ids.view(batch_size, n_beams, seq_len)
            
            # (N, n_beams, S) -> (N, n_beams, n_beams, S)
            tgt_ids = tgt_ids.unsqueeze(2).repeat(1, 1, n_beams, 1)
            
            tgt_ids[:, :, :, i+1] = torch.where(has_eos.unsqueeze(-1), tokenizer.pad_id(), candidates)
            
            tgt_log_likehood = tgt_log_likehood.unsqueeze(2).repeat(1, 1, n_beams)
            
            # ブロードキャストで複製が増えるの回避
            pad_porba = torch.tensor([0.0] + (n_beams-1)*[float('inf')], device=device)
            tgt_log_likehood += torch.log(torch.where(has_eos.unsqueeze(-1), pad_porba, proba))
            
            tgt_log_likehood, top_idx = tgt_log_likehood.view(batch_size,n_beams*n_beams).topk(n_beams)

            tgt_ids = tgt_ids.view(batch_size, n_beams*n_beams, seq_len)
            top_idx = top_idx.unsqueeze(-1).expand(-1, -1, seq_len)

            tgt_ids = tgt_ids.gather(1, top_idx)
            
        has_eos = torch.any((tgt_ids == tokenizer.eos_id()), dim=-1)
        
        if torch.all(has_eos):
            break

    # 対数尤度の一番大きいもの   
    pred_ids = tgt_ids[:, 0, :]
        
    if not is_batched:
        pred_ids.squeeze_(0)
        
    return pred_ids



if __name__ == '__main__':
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('.tokenizer/en_de_bpe_37000.model')
    
    model = TransformerModel(tokenizer.piece_size(), tokenizer.piece_size()).eval().to(DEVICE)
    model.load_state_dict(torch.load('transformer-weight-ls.pth', weights_only=True))
    
    ds = WMT14_DE_EN('test', tokenizer)
    loader = DataLoader(ds, batch_size=64)
    

    metric = BLEUScore(n_gram=4)
    
    with torch.no_grad():
    
        for src, tgt, label in tqdm(loader):
            
            src, label = src.to(DEVICE), label.to(DEVICE)

            pred = beam_search(model, tokenizer, src)

            if pred.dim() == 1:
                pred.unsqueeze_(0)
                label.unsqueeze_(0)
                
            candidates = [tokenizer.DecodeIds(x.tolist()) for x in pred]
            references = [[tokenizer.DecodeIds(x.tolist())] for x in label]
            metric.update(candidates, references)
        
    print('BLEU score:', metric.compute().item())
            