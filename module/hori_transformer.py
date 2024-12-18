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

class MySingleHead(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 head: int):
        super().__init__(MySingleHead, self).__init__()

        self.dmodel = out_features

        self.W_Q = nn.Parameter(
            torch.empty(out_features,self.dmodel)   
        )

        self.W_K = nn.Parameter(
            torch.empty(out_features,self.dmodel)   
        )

        self.W_V = nn.Parameter(
            torch.empty(out_features,self.dmodel)   
        )

        init.kaiming_uniform_(self.W_Q, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_K, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_V, a=math.sqrt(5))

        



    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.Q = torch.matmul(input, self.W_Q)
        self.K = torch.matmul(input, self.W_K)
        self.V = torch.matmul(input, self.W_V)

        self.KT = torch.transpose(self.K,0,1)

        x = torch.matmul(self.W_Q,self.KT)
        x = 
        return x


if __name__ == '__main__':
    
    x = torch.randn(15, 10)
    
    Single_Attention = nn.Transformer()
    
    Single_Attention = MySingleHead(in_features=x.size()[0], out_features=x.size()[1], head=1)
    
    out = Single_Attention(x)
    
    #out.mean().backward()
    
    #print(linear.bias.grad)
    