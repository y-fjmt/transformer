import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TransformerModel
from dataset import WMT14_DE_EN
from tokenizer import WMT14Tokenizer

from accelerate import Accelerator


class CFG:
    
    DEBUG = False
    
    BS = 256
    IS_SHUFFLE = True
    N_WORKERS = 10
    
    SEQ_LEN = 128
    
    LR = 1e-3
    EPOCHS = 5
    SAVE_AS = 'transformer-weight.pth'
    N_HEADS = 8


if __name__ == '__main__':
    
    accelerator = Accelerator()
    
    src_tokenizer = WMT14Tokenizer(lang='en', is_debug=CFG.DEBUG, hide_pbar=(not accelerator.is_local_main_process), max_length=CFG.SEQ_LEN)
    tgt_tokenizer = WMT14Tokenizer(lang='de', is_debug=CFG.DEBUG, hide_pbar=(not accelerator.is_local_main_process), max_length=CFG.SEQ_LEN)
    
    PAD_IDX = src_tokenizer.word_to_id[src_tokenizer.pad_token]
    
    trn_ds = WMT14_DE_EN('train', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    val_ds = WMT14_DE_EN('val', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    
    trn_loader = DataLoader(trn_ds, CFG.BS, CFG.IS_SHUFFLE, num_workers=CFG.N_WORKERS)
    val_loader = DataLoader(val_ds, CFG.BS, CFG.IS_SHUFFLE, num_workers=CFG.N_WORKERS)
    
    model = TransformerModel(src_tokenizer.vocab_size(), tgt_tokenizer.vocab_size())
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)
    sheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.EPOCHS)
    
    # (BS*num_heads, seq, seq)
    src_msk = torch.zeros((CFG.SEQ_LEN, CFG.SEQ_LEN)).unsqueeze(0).expand(CFG.BS*CFG.N_HEADS, -1, -1)
    tgt_msk = nn.Transformer.generate_square_subsequent_mask(CFG.SEQ_LEN).unsqueeze(0).expand(CFG.BS*CFG.N_HEADS, -1, -1)
    
    trn_loader, val_loader = accelerator.prepare(trn_loader, val_loader)
    model, optimizer, sheduler = accelerator.prepare(model, optimizer, sheduler)
    src_msk, tgt_msk = accelerator.prepare(src_msk, tgt_msk)
    
    for epoch in range(1, CFG.EPOCHS+1):
        
        trn_loss, val_loss = 0, 0
        
        # train
        pbar = tqdm(trn_loader, disable=(not accelerator.is_local_main_process))
        pbar.set_description(f'[Epoch {epoch:03d}/trn]')
        
        for iter_idx, (src, tgt, label) in enumerate(pbar):
            
            src, tgt, label = accelerator.prepare(src, tgt, label)
            
            # (N, seq)
            src_padding_msk = (src == PAD_IDX).float()
            tgt_padding_msk = (tgt == PAD_IDX).float()
            
            # (seq_len, batch_size) -> (seq_len, batch_size, tgt_vocab)
            logits = model(src, tgt, src_mask=src_msk, tgt_mask=tgt_msk, 
                           src_padding_mask=src_padding_msk, tgt_padding_mask=tgt_padding_msk, 
                           memory_key_padding_mask=src_padding_msk)
            
            
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
            
            pbar = tqdm(val_loader, disable=(not accelerator.is_local_main_process))
            pbar.set_description(f'[Epoch {epoch:03d}/val]')
            
            for iter_idx, (src, tgt, label) in enumerate(pbar):
            
                src, tgt, label = accelerator.prepare(src, tgt, label)
            
                # (N, seq)
                src_padding_msk = (src == PAD_IDX).float()
                tgt_padding_msk = (tgt == PAD_IDX).float()
                
                # (seq_len, batch_size) -> (seq_len, batch_size, tgt_vocab)
                logits = model(src, tgt, src_mask=src_msk, tgt_mask=tgt_msk, 
                            src_padding_mask=src_padding_msk, tgt_padding_mask=tgt_padding_msk, 
                            memory_key_padding_mask=src_padding_msk)
                
                # (batch_size, seq_len, tgt_vocab) -> (batch_size * seq_len, tgt_vocab)
                tgt_vocab_size = tgt_tokenizer.vocab_size()
                logits = logits.contiguous().view(-1, tgt_vocab_size)
                
                # (batch_size, seq_len) -> (batch_size * seq_len)
                label = label.view(-1)
                
                loss = loss_fn(logits, label)
                
                val_loss += loss
                pbar.set_postfix({'validation_loss': val_loss.item() / ((iter_idx+1)*CFG.BS)})
        
    
    accelerator.save(accelerator.unwrap_model(model), CFG.SAVE_AS)