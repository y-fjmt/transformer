import math
import torch
from torch import nn
from torch.nn import functional as F, init

class MyLinear(nn.Module):
    
    def __init__(self,
                 in_features: int, out_features: int,
                 bias: bool = False
                ) -> None:
        super(MyLinear, self).__init__()
        
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features)   
        )
        
        self.bias = nn.Parameter(
            torch.empty(out_features)
        )
        
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(input, self.weight)
        x = x + self.bias
        return x
    

if __name__ == '__main__':
    
    x = torch.randn(15, 10)
    
    linear = nn.Linear(in_features=10, out_features=8)
    linear = MyLinear(in_features=10, out_features=8)
    
    out = linear(x)
    
    out.mean().backward()
    
    print(linear.bias.grad)
    
    