import os
import torch

class CFG:
    
    DEBUG = False
    
    BS = 256
    IS_SHUFFLE = True
    N_WORKERS = 4
    
    SEQ_LEN = 128
    
    LR = 1e-3
    LABEL_SMOOTHING = 0.1
    EPOCHS = 30
    SAVE_AS = 'translate-de-en-true-.pth'
    EXP_NAME = f'EN-DE Translate Model, w/grad_clip'
    
    CLIP_VALUE = 0.5
    
    BELU_N_GRAM = 4
    N_BEAM = 4