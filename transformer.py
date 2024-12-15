import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TransformerModel
from dataset import WMT14_DE_EN
from tokenizer import WMT14Tokenizer
    
class CFG:
    DEBUG = False
    DEVICE = 'mps'
    
    BS = 32
    IS_SHUFFLE = True
    N_WORKERS = 4
    
    LR = 1e-3
    EPOCHS = 1
    SAVE_DIR = 'weight.pth'


if __name__ == '__main__':
    
    src_tokenizer = WMT14Tokenizer(lang='en', is_debug=CFG.DEBUG)
    tgt_tokenizer = WMT14Tokenizer(lang='de', is_debug=CFG.DEBUG)
    
    trn_ds = WMT14_DE_EN('train', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    val_ds = WMT14_DE_EN('val', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    tst_ds = WMT14_DE_EN('test', src_tokenizer, tgt_tokenizer,is_debug=CFG.DEBUG)
    
    trn_loader = DataLoader(trn_ds, CFG.BS, CFG.IS_SHUFFLE, num_workers=CFG.N_WORKERS)
    val_loader = DataLoader(val_ds, CFG.BS, CFG.IS_SHUFFLE, num_workers=CFG.N_WORKERS)
    tst_loader = DataLoader(tst_ds, CFG.BS, CFG.IS_SHUFFLE, num_workers=CFG.N_WORKERS)
    
    model = TransformerModel(src_tokenizer.vocab_size(), tgt_tokenizer.vocab_size()).to(CFG.DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.LR)
    sheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, CFG.EPOCHS)
    
    
    for epoch in range(1, CFG.EPOCHS+1):
        
        trn_loss, val_loss = 0, 0
        
        # train
        pbar = tqdm(trn_loader)
        pbar.set_description(f'[Epoch {epoch:03d}/trn]')
        
        for iter_idx, (src, tgt) in enumerate(pbar):
            
            src, tgt = src.to(CFG.DEVICE), tgt.to(CFG.DEVICE)
            outputs = model(src=src, tgt=tgt)
            
            tgt_emb = model.tgt_embedding(tgt)
            loss = loss_fn(outputs, tgt_emb)
            
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
            
            for iter_idx, (src, tgt) in enumerate(pbar):
            
                src, tgt = src.to(CFG.DEVICE), tgt.to(CFG.DEVICE)
                outputs = model(src=src, tgt=tgt)
                tgt_emb = model.tgt_embedding(tgt)
                loss = loss_fn(outputs, tgt_emb)
                val_loss += loss
                pbar.set_postfix({'validation_loss': val_loss.item() / ((iter_idx+1)*CFG.BS)})
    
    
    model = model.to('cpu').eval()
    torch.save(model.state_dict(), CFG.SAVE_DIR)
    # # test
    # with torch.no_grad():
            
    #     pbar = tqdm(tst_loader)
    #     for src, tgt in pbar:
        
    #         src, tgt = src.to(CFG.DEVICE), tgt.to(CFG.DEVICE)
    #         outputs = model(src=src, tgt=tgt)
    #         loss = loss_fn(outputs, tgt)