import math
import torch
from torch import nn
from torch.nn import functional as F, init

# (
#     d_model: int = 512
#     nhead: int = 8
#     num_encoder_layers: int = 6
#     num_decoder_layers: int = 6
#     dim_feedforward: int = 2048
#     dropout: float = 0.1
#     activation: str | ((Tensor) -> Tensor) = F.relu, custom_encoder: Any | None = None
#     custom_decoder: Any | None = None
#     layer_norm_eps: float = 0.00001
#     batch_first: bool = False
#     norm_first: bool = False, 
#     bias: bool = True, 
#     device: Any | None = None, 
#     dtype: Any | None = None
# ) -> Transformer


class fukui_attention(nn.Module):
    def __init__(self, 
                #単語のうめこみちょう
                d_model: int,
                seq: int,
                key_dim: int,
                out_feature: int,
                bias: bool = False
                ):
        super().__init__()
        
        self.key_weight = nn.Parameter(
            torch.empty(seq, d_model)
        )
        
        self.query_weight = nn.Parameter(
            torch.empty(in_feature, out_feature)
        )
        
        self.value_weight = nn.Parameter(
            torch.empty(in_feature, out_feature)
        )
        
             
        self.bias = nn.Parameter(
            torch.empty(out_feature)
        )
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            
            
        
        
    def forward(self, input: torch.Tensor):
        