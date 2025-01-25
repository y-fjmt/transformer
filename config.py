import os
import torch

class CFG:
    
    DEBUG = False
    
    BS = 128
    IS_SHUFFLE = True
    N_WORKERS = 10
    
    SEQ_LEN = 128
    
    LR = 1e-3
    LABEL_SMOOTHING = 0.1
    EPOCHS = 15
    SAVE_AS = 'transformer-weight.pth'
    EXP_NAME = f'EN-DE Translate Model, BS:{BS}'