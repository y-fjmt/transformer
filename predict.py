import torch
from torch import nn
from tqdm import tqdm
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import TransformerModel
from dataset import WMT14_DE_EN, Collator

from torcheval.metrics import BLEUScore
from accelerate import Accelerator

from config import CFG

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def beam_search(
        model: TransformerModel,
        tokenizer: spm.SentencePieceProcessor,
        src_ids: torch.Tensor, 
        n_beams: int = 4,
        device: torch.DeviceObjType = DEFAULT_DEVICE
    ) -> torch.Tensor:
    """ビームサーチで翻訳

    Args:
        model (TransformerModel): 学習済みのモデル
        tokenizer (spm.SentencePieceProcessor): トークナイザ
        src_ids (torch.Tensor): 入力の単語列 (S, ) or (N, S)
        n_beams (int, optional): ビーム数. Defaults to 4.

    Returns:
        torch.Tensor: 対数尤度の最も高い結果 (S, ) or (N, S)
    """
    
    assert src_ids.device == next(model.parameters()).device, \
        '`src_ids` and `model` must be same device.'
    
    assert src_ids.dim() == 1 or src_ids.dim() == 2, \
        f'`src_ids` must be 1D or 2D tensor, got {src_ids.dim()}D tensor.'
    is_batched = (src_ids.dim() == 2)
    if not is_batched:
        src_ids = src_ids.unsqueeze(0)
        
    N, S = src_ids.shape
    
    src_msk = torch.zeros(S, S, device=device)
    src_padding_msk = (src_ids == tokenizer.pad_id()).float()
    
    tgt_ids = torch.full_like(src_ids, tokenizer.pad_id(), device=device)
    tgt_ids[:, 0] = tokenizer.bos_id()
    tgt_msk = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        
    # エンコーダの計算
    memory = model.enc_forward(src_ids, src_msk, src_padding_msk)
    
    tgt_log_likehood = torch.full((N, n_beams), 0.0, device=device)
    has_eos = torch.full((N, n_beams), False, device=device)
    
    # デコーダを再帰的に計算
    for i in range(S-1):
        
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
            memory = memory.unsqueeze(1).repeat(1, n_beams, 1, 1).view(N*n_beams, S, -1)
            src_padding_msk = src_padding_msk.unsqueeze(1).repeat(1, n_beams, 1).view(N*n_beams, S)
            
        else:
            
            # (N, S)の形にする
            tgt_ids = tgt_ids.view(-1, S)
            
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
            candidates = candidates.view(N, n_beams, n_beams)
            proba = proba.view(N, n_beams, n_beams)
            
            # (N*n_beams, S) -> (N, n_beams, S)
            tgt_ids = tgt_ids.view(N, n_beams, S)
            
            # (N, n_beams, S) -> (N, n_beams, n_beams, S)
            tgt_ids = tgt_ids.unsqueeze(2).repeat(1, 1, n_beams, 1)
            
            tgt_ids[:, :, :, i+1] = torch.where(has_eos.unsqueeze(-1), tokenizer.pad_id(), candidates)
            
            tgt_log_likehood = tgt_log_likehood.unsqueeze(2).repeat(1, 1, n_beams)
            
            # ブロードキャストで複製が増えるの回避
            pad_porba = torch.tensor([0.0] + (n_beams-1)*[float('inf')], device=device)
            tgt_log_likehood += torch.log(torch.where(has_eos.unsqueeze(-1), pad_porba, proba))
            
            tgt_log_likehood, top_idx = tgt_log_likehood.view(N,n_beams*n_beams).topk(n_beams)

            tgt_ids = tgt_ids.view(N, n_beams*n_beams, S)
            top_idx = top_idx.unsqueeze(-1).expand(-1, -1, S)

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
    
    accelerator = Accelerator()
    
    metric = BLEUScore(n_gram=4)
    model = TransformerModel(tokenizer.piece_size(), tokenizer.pad_id()).eval()
    model.load_state_dict(torch.load('translate-de-en-bs-64_long.pth', weights_only=True))
    
    collate_fn = Collator(tokenizer.pad_id())
    ds = WMT14_DE_EN('test', tokenizer)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
    
    model, loader = accelerator.prepare(model, loader)
    
    with torch.no_grad():
        
        pbar = tqdm(loader, disable=not accelerator.is_local_main_process)
        
        for src, tgt, label in pbar:
            
            pred = beam_search(model.module, tokenizer, src, device=accelerator.device)
            
            
            N, S = src.shape
            pad_idx = tokenizer.pad_id()
            pad = torch.full((N, CFG.SEQ_LEN - S), pad_idx, device=accelerator.device)
            
            pred  = torch.cat([pred, pad], dim=1)
            label = torch.cat([label, pad], dim=1)
            
            pred_gathered = accelerator.gather_for_metrics(pred)
            label_gathered = accelerator.gather_for_metrics(label)
                
            candidates = [tokenizer.DecodeIds(x.tolist()) for x in pred_gathered]
            references = [[tokenizer.DecodeIds(x.tolist())] for x in label_gathered]
            
            metric.update(candidates, references)
                
    
    belu = metric.compute().item() * 100
    if accelerator.is_local_main_process:
        print('BELU:', belu)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
                