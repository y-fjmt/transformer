import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import TransformerModel, create_mask
from dataset import WMT14_DE_EN
from tokenizer import WMT14Tokenizer


class CFG:
    DEBUG = False
    DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    BS = 64
    IS_SHUFFLE = True
    N_WORKERS = 4
    
    LR = 1e-3
    EPOCHS = 50
    SAVE_AS = 'transformer-weight.pth'


if __name__ == '__main__':
    
    # DDP initialization
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    CFG.DEVICE = rank % torch.cuda.device_count()
    
    src_tokenizer = WMT14Tokenizer(lang='en', is_debug=CFG.DEBUG, hide_pbar=(rank != 0))
    tgt_tokenizer = WMT14Tokenizer(lang='de', is_debug=CFG.DEBUG, hide_pbar=(rank != 0))
    
    PAD_IDX = src_tokenizer.word_to_id[src_tokenizer.pad_token]
    
    trn_ds = WMT14_DE_EN('train', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    val_ds = WMT14_DE_EN('val', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    
    trn_sampler = torch.utils.data.distributed.DistributedSampler(
                        trn_ds, 
                        num_replicas=dist.get_world_size(), 
                        rank=dist.get_rank(),
                        shuffle=CFG.IS_SHUFFLE)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
                        val_ds, 
                        num_replicas=dist.get_world_size(), 
                        rank=dist.get_rank(),
                        shuffle=CFG.IS_SHUFFLE)
    
    trn_loader = DataLoader(trn_ds, CFG.BS, sampler=trn_sampler, num_workers=CFG.N_WORKERS)
    val_loader = DataLoader(val_ds, CFG.BS, sampler=val_sampler, num_workers=CFG.N_WORKERS)
    
    model = TransformerModel(src_tokenizer.vocab_size(), tgt_tokenizer.vocab_size()).to(CFG.DEVICE)
    ddp_model = DDP(model, device_ids=[CFG.DEVICE])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(ddp_model.parameters(), lr=CFG.LR)
    sheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.EPOCHS)
    
    
    for epoch in range(1, CFG.EPOCHS+1):
        
        trn_loss, val_loss = 0, 0
        
        # train
        pbar = tqdm(trn_loader, disable=(rank != 0))
        pbar.set_description(f'[Epoch {epoch:03d}/trn]')
        
        for iter_idx, (src, tgt, label) in enumerate(pbar):
            
            src, tgt, label = src.to(CFG.DEVICE), tgt.to(CFG.DEVICE), label.to(CFG.DEVICE)
            
            # (batch_size, seq_len) -> (seq_len, batch_size)
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            
            
            # prepare masks
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, PAD_IDX)
            src_mask, tgt_mask = src_mask.to(CFG.DEVICE), tgt_mask.to(CFG.DEVICE)
            src_padding_mask, tgt_padding_mask = src_padding_mask.to(CFG.DEVICE), tgt_padding_mask.to(CFG.DEVICE)
            
            # (seq_len, batch_size) -> (seq_len, batch_size, tgt_vocab)
            logits = ddp_model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            # (seq_len, batch_size, tgt_vocab) -> (batch_size, seq_len, tgt_vocab)
            logits = logits.transpose(0, 1)
            
            # (batch_size, seq_len, tgt_vocab) -> (batch_size * seq_len, tgt_vocab)
            tgt_vocab_size = tgt_tokenizer.vocab_size()
            logits = logits.contiguous().view(-1, tgt_vocab_size)
            
            # (batch_size, seq_len) -> (batch_size * seq_len)
            label = label.view(-1)
            
            loss = loss_fn(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sheduler.step()
            
            trn_loss += loss
            pbar.set_postfix({'train_loss': trn_loss.item() / ((iter_idx+1)*CFG.BS)})
            
            
        # evaluation
        with torch.no_grad():
            
            pbar = tqdm(val_loader, disable=(rank != 0))
            pbar.set_description(f'[Epoch {epoch:03d}/val]')
            
            for iter_idx, (src, tgt, label) in enumerate(pbar):
            
                src, tgt, label = src.to(CFG.DEVICE), tgt.to(CFG.DEVICE), label.to(CFG.DEVICE)
            
                # (batch_size, seq_len) -> (seq_len, batch_size)
                src = src.transpose(0, 1)
                tgt = tgt.transpose(0, 1)
                
                
                # prepare masks
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt, PAD_IDX)
                src_mask, tgt_mask = src_mask.to(CFG.DEVICE), tgt_mask.to(CFG.DEVICE)
                src_padding_mask, tgt_padding_mask = src_padding_mask.to(CFG.DEVICE), tgt_padding_mask.to(CFG.DEVICE)
                
                # (seq_len, batch_size) -> (seq_len, batch_size, tgt_vocab)
                logits = ddp_model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                
                # (seq_len, batch_size, tgt_vocab) -> (batch_size, seq_len, tgt_vocab)
                logits = logits.transpose(0, 1)
                
                # (batch_size, seq_len, tgt_vocab) -> (batch_size * seq_len, tgt_vocab)
                tgt_vocab_size = tgt_tokenizer.vocab_size()
                logits = logits.contiguous().view(-1, tgt_vocab_size)
                
                # (batch_size, seq_len) -> (batch_size * seq_len)
                label = label.view(-1)
                
                loss = loss_fn(logits, label)
                
                val_loss += loss
                pbar.set_postfix({'validation_loss': val_loss.item() / ((iter_idx+1)*CFG.BS)})
        
    # save model
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CFG.SAVE_AS)
        dist.destroy_process_group()