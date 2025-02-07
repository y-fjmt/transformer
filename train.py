import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm
from accelerate import Accelerator
from torcheval.metrics import BLEUScore
from tqdm import tqdm

from model import TransformerModel
from dataset import WMT14_DE_EN, Collator
from predict import beam_search
from sheduler import CustomLRScheduler
from config import CFG



if __name__ == '__main__':
    
    accelerator = Accelerator()
    writer = SummaryWriter(log_dir=CFG.TFB_DIR)
    
    if accelerator.is_local_main_process:
        print('='*20, CFG.EXP_NAME ,'='*20)
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('.tokenizer/en_de_bpe_37000.model')
    
    PAD_IDX = tokenizer.pad_id()
    
    collate_fn = Collator(PAD_IDX)
    
    trn_ds = WMT14_DE_EN('train', tokenizer, is_debug=CFG.DEBUG)
    val_ds = WMT14_DE_EN('val', tokenizer, is_debug=CFG.DEBUG)
    
    trn_loader = DataLoader(trn_ds, CFG.BS, CFG.IS_SHUFFLE, collate_fn=collate_fn, num_workers=CFG.N_WORKERS)
    val_loader = DataLoader(val_ds, CFG.BS_VAL, CFG.IS_SHUFFLE, collate_fn=collate_fn,num_workers=CFG.N_WORKERS)
    
    model = TransformerModel(tokenizer.piece_size(), tokenizer.pad_id())
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=CFG.LABEL_SMOOTHING)
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR, betas=[0.9, 0.98], eps=1e-9)
    sheduler = CustomLRScheduler(optimizer,d_model=512, warmup_steps=4000)
    
    # send them to each device
    trn_loader, val_loader = accelerator.prepare(trn_loader, val_loader)
    model, optimizer, sheduler = accelerator.prepare(model, optimizer, sheduler)
    
    for epoch in range(1, CFG.EPOCHS+1):
        
        trn_loss, val_loss = 0, 0
        
        # train
        model.train()
        pbar = tqdm(trn_loader, disable=(not accelerator.is_local_main_process))
        
        for iter_idx, (src, tgt, label) in enumerate(pbar):
            
            optimizer.zero_grad()
            
            N, S = src.shape
            src_msk = torch.zeros((S, S), device=accelerator.device)
            tgt_msk = nn.Transformer.generate_square_subsequent_mask(S, accelerator.device)
            src_padding_msk = (src == PAD_IDX).float()
            tgt_padding_msk = (tgt == PAD_IDX).float()
            
            
            logits = model(src, tgt, 
                           src_mask=src_msk, tgt_mask=tgt_msk, 
                           src_padding_mask=src_padding_msk, tgt_padding_mask=tgt_padding_msk, 
                           memory_key_padding_mask=src_padding_msk)
            
            
            # flatten to compute loss
            logits = logits.contiguous().view(-1, tokenizer.piece_size())
            loss = loss_fn(logits, label.view(-1))
            
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), CFG.CLIP_VALUE)
            optimizer.step()
            sheduler.step()
            
            trn_loss += loss.item()
            pbar.set_postfix({'train_loss': loss.item()})
            
            
        # evaluation
        model.eval()
        with torch.no_grad():
            
            pbar = tqdm(val_loader, disable=(not accelerator.is_local_main_process))
            
            for iter_idx, (src, tgt, label) in enumerate(pbar):
            
                N, S = src.shape
                src_msk = torch.zeros((S, S), device=accelerator.device)
                tgt_msk = nn.Transformer.generate_square_subsequent_mask(S, accelerator.device)
                src_padding_msk = (src == PAD_IDX).float()
                tgt_padding_msk = (tgt == PAD_IDX).float()
                
                # (seq_len, batch_size) -> (seq_len, batch_size, tgt_vocab)
                logits = model(src, tgt, 
                               src_mask=src_msk, tgt_mask=tgt_msk, 
                               src_padding_mask=src_padding_msk, tgt_padding_mask=tgt_padding_msk, 
                               memory_key_padding_mask=src_padding_msk)
                
                # (batch_size, seq_len, tgt_vocab) -> (batch_size * seq_len, tgt_vocab)
                tgt_vocab_size = tokenizer.piece_size()
                logits = logits.contiguous().view(-1, tgt_vocab_size)
                
                # (batch_size, seq_len) -> (batch_size * seq_len)
                label = label.view(-1)
                
                loss = loss_fn(logits, label)
                
                val_loss += loss.item()
                pbar.set_postfix({'validation_loss': loss.item()})
                
            
            # compute BLEU score using beamsearch
            metric = BLEUScore(n_gram=CFG.BELU_N_GRAM)
            pbar = tqdm(val_loader, disable=(not accelerator.is_local_main_process))
            
            for src, tgt, label in pbar:

                pred = beam_search(model.module, tokenizer, src, device=accelerator.device)
                
                # padding to gather
                N, S = src.shape
                pad_idx = tokenizer.pad_id()
                pad = torch.full((N, CFG.SEQ_LEN - S), pad_idx, device=accelerator.device)
                pred  = torch.cat([pred, pad], dim=1)
                label = torch.cat([label, pad], dim=1)
                
                
                pred_gathered = accelerator.gather_for_metrics(pred)
                label_gathered = accelerator.gather_for_metrics(label)
                    
                candidates = [tokenizer.DecodeIds(x.tolist()) for x in pred_gathered]
                references = [[tokenizer.DecodeIds(x.tolist())] for x in label_gathered]
                
                try:
                    # 生成された文が短すぎるとBELUが生成できない
                    metric.update(candidates, references)
                except ValueError:
                    pass
            
            belu = metric.compute().item() * 100
        
        if accelerator.is_local_main_process:
            print('[Epoch {:03d}/{:03d}]: train_loss: {:.4f}, valid_loss: {:.4f}, BELU: {:.4f}'.format(
                epoch ,CFG.EPOCHS, 
                trn_loss/len(trn_loader), val_loss/len(val_loader), belu
            ))
            print('-'*80)
            
            writer.add_scalar("train loss", trn_loss/len(trn_loader), epoch)
            writer.add_scalar("validation loss", val_loss/len(val_loader), epoch)
            writer.add_scalar("BELU score", belu, epoch)
    
    
    # finally, save trained model as state_dict
    if accelerator.is_main_process:
        torch_model = accelerator.unwrap_model(model).to('cpu')
        torch.save(torch_model.state_dict(), CFG.SAVE_AS)
        writer.close()
        
    
    # destroy ProcessGroupNCCL
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()