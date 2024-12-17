import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TransformerModel, create_mask
from dataset import WMT14_DE_EN
from tokenizer import WMT14Tokenizer


class CFG:
    DEBUG = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    
    BS = 32
    IS_SHUFFLE = True
    N_WORKERS = 4
    
    LR = 1e-3
    EPOCHS = 10
    SAVE_AS = 'transformer-weight.pth'


if __name__ == '__main__':
    
    src_tokenizer = WMT14Tokenizer(lang='en', is_debug=CFG.DEBUG, hide_pbar=False)
    tgt_tokenizer = WMT14Tokenizer(lang='de', is_debug=CFG.DEBUG, hide_pbar=False)
    
    PAD_IDX = src_tokenizer.word_to_id[src_tokenizer.pad_token]
    
    trn_ds = WMT14_DE_EN('train', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    val_ds = WMT14_DE_EN('val', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    
    trn_loader = DataLoader(trn_ds, CFG.BS, CFG.IS_SHUFFLE, num_workers=CFG.N_WORKERS)
    val_loader = DataLoader(val_ds, CFG.BS, CFG.IS_SHUFFLE, num_workers=CFG.N_WORKERS)
    
    model = TransformerModel(src_tokenizer.vocab_size(), tgt_tokenizer.vocab_size()).to(CFG.DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)
    sheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.EPOCHS)
    
    model, optimizer, sheduler, trn_loader, val_loader
    
    for epoch in range(1, CFG.EPOCHS+1):
        
        trn_loss, val_loss = 0, 0
        
        # train
        pbar = tqdm(trn_loader)
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
            logits = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
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
            
            pbar = tqdm(val_loader)
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
                logits = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                
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
    model = model.to('cpu')
    torch.save(model.state_dict(), CFG.SAVE_AS)