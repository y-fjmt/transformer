import sys

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any


class MultiHeadAttention(nn.Module):
    
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            kdim: int | None = None,
            vdim: int | None = None,
            batch_first: bool = False,
            device: Any | None = None,
            dtype: Any | None = None
        ) -> None:
        
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.d_k = kdim if kdim is not None else int(embed_dim / num_heads)
        self.d_v = vdim if vdim is not None else int(embed_dim / num_heads)
        
        self.W_q = nn.Linear(embed_dim, self.d_k*num_heads, bias)
        self.W_k = nn.Linear(embed_dim, self.d_k*num_heads, bias)
        self.W_v = nn.Linear(embed_dim, self.d_v*num_heads, bias)
        self.W_o = nn.Linear(self.d_v*num_heads, embed_dim, bias)
        
        # print('Wq:', self.W_q.weight.transpose(0, 1).shape)
        # print('Wk:', self.W_k.weight.transpose(0, 1).shape)
        # print('Wv:', self.W_v.weight.transpose(0, 1).shape)
        # print('Wo:', self.W_o.weight.transpose(0, 1).shape)
        
        
    def forward(
            self, 
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
            need_weights: bool = True,
            attn_mask: torch.Tensor | None = None,
            average_attn_weights: bool = True,
            is_causal: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
        
        if not query.dim() == key.dim() == value.dim():
            raise ValueError('Dimensions are not matched between query, key and value')
        
        is_batched = query.dim() == 3
        
        # print('query:', query.shape)
        
        Q: torch.Tensor = self.W_q(query)
        K: torch.Tensor = self.W_k(key)
        V: torch.Tensor = self.W_v(value)
        
        # print('Q:', Q.shape)
        # print(Q)
        # print(V)
        # print('K:', K.shape)
        # print('V:', V.shape)
        
        scale = math.sqrt(self.d_k)
        attn = []
        
        for i in range(self.num_heads):
            
            if is_batched:
                Q_i = Q[:, :, i*self.d_k:(i+1)*self.d_k]
                K_i = K[:, :, i*self.d_k:(i+1)*self.d_k]
                V_i = V[:, :, i*self.d_v:(i+1)*self.d_v]
            else:
                Q_i = Q[:, i*self.d_k:(i+1)*self.d_k]
                K_i = K[:, i*self.d_k:(i+1)*self.d_k]
                V_i = V[:, i*self.d_v:(i+1)*self.d_v]
                
            
            K_i_t = K_i.transpose(-2, -1)
            
            # print(Q_i.shape)
            # print(K_i_t.shape)
            
            
            attn_mat = torch.matmul(Q_i, K_i_t) / scale
            
            attn_mat = F.softmax(attn_mat, dim=-1)

            attn.append(torch.matmul(attn_mat, V_i))
        
        attn = torch.cat(attn, dim=-1)
        
        # print('attn:', attn.shape)
        # print(attn)
        
        # print('self.W_o.bias:', self.W_o.bias.shape)
        # print(self.W_o.bias[:10])
        
        mh_attn = self.W_o(attn)
        
        # print('mh_attn:', mh_attn.shape)
        # print(mh_attn)
        
        return mh_attn, None


    def load_weights_from_torch(self, mha: nn.MultiheadAttention) -> None:
        
        if mha._qkv_same_embed_dim:
            # print(self.W_q.weight.shape)
            # print(mha.in_proj_weight[0*mha.embed_dim:1*mha.embed_dim, :].shape)
            
            self.W_q.weight = nn.Parameter(
                mha.in_proj_weight.chunk(3)[0]
            )
            self.W_k.weight = nn.Parameter(
                mha.in_proj_weight.chunk(3)[1]
            )
            self.W_v.weight = nn.Parameter(
                mha.in_proj_weight.chunk(3)[2]
            )
            
            if mha.in_proj_bias is not None:
                # print('bias:', mha.in_proj_bias.shape)
                self.W_q.bias = nn.Parameter(
                    mha.in_proj_bias.chunk(3)[0]
                )
                self.W_k.bias = nn.Parameter(
                    mha.in_proj_bias.chunk(3)[1]
                )
                self.W_v.bias = nn.Parameter(
                    mha.in_proj_bias.chunk(3)[2]
                )   
            
            self.W_o.weight = nn.Parameter(
                mha.out_proj.weight
            )
            
            # print(mha.out_proj.bias)
            
            if mha.out_proj.bias is not None:
                
                # print('mha.out_proj.bias:', mha.out_proj.bias.shape)
                
                self.W_o.bias = nn.Parameter(
                    mha.out_proj.bias
                ) 
                
        else:
            raise NotImplementedError