import os
import torch

class CFG:
    
    DEBUG = False
    
    TFB_DIR = 'log/exp_cliped/'
    
    BS = 128
    IS_SHUFFLE = True
    N_WORKERS = 8
    
    SEQ_LEN = 128
    
    LR = 1e-3
    LABEL_SMOOTHING = 0.1
    EPOCHS = 15
    SAVE_AS = 'translate-de-en-cliped.pth'
    EXP_NAME = f'save to {TFB_DIR}'
    
    CLIP_VALUE = 0.5
    
    BELU_N_GRAM = 4
    N_BEAM = 4
    BS_VAL = BS // N_BEAM
    