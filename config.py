import os
import torch

class CFG:
    
    DEBUG = False
    
    TFB_DIR = 'log/exp_seq_rev_1/'
    
    BS = 64
    IS_SHUFFLE = False
    N_WORKERS = 10
    
    SEQ_LEN = 128
    
    LR = 1e-3
    LABEL_SMOOTHING = 0.1
    EPOCHS = 30
    SAVE_AS = 'translate-de-en-seq-rev-1.pth'
    EXP_NAME = f'save to {TFB_DIR}'
    
    CLIP_VALUE = 0.5
    
    BELU_N_GRAM = 4
    N_BEAM = 4
    BS_VAL = BS // N_BEAM
    