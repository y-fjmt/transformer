import torch
from torch import nn
import torch.nn.functional as F
from mha import MultiHeadAttention
import pytest

torch.manual_seed(42)

class TestMultiheadAttention:
    
    @staticmethod
    def _prepare(
            d_model: int = 512, 
            num_heads: int = 8, 
            bias: bool = True
        ) -> tuple[MultiHeadAttention, nn.MultiheadAttention]:
        
        _attn = nn.MultiheadAttention(d_model, num_heads, bias=bias, batch_first=True)
        attn = MultiHeadAttention(d_model, num_heads, bias=bias)
        attn.load_weights_from_torch(_attn)
        
        return attn, _attn
    
    
    def _test(
        self,
        embed_dim: int,
        num_heads: int,
        bias = True,
    ) -> None:
        self._prepare()
        
    
    @pytest.mark.parametrize("bias", [(True,), (False,)])
    def test_args(
            self, 
            bias:bool
        ):
        seq_len, d_model, num_heads = 128, 512, 8
        mha, _mha = self._prepare(d_model, num_heads, bias=bias)
        
        k = torch.randn(1, seq_len, d_model)
        q = torch.randn(1, seq_len, d_model)
        v = torch.randn(1, seq_len, d_model)
        
        # forward test
        out, out_w  = mha(k, q, v)
        _out, _out_w = _mha(k, q, v)
        
        assert torch.allclose(out, _out), 'Outputs are not same.'
        
        # backward test
        out.mean().backward()
        _out.mean().backward()
        
        wq_grad, _wq_grad = mha.W_q.weight.grad, _mha.in_proj_weight.grad.chunk(3)[0]
        wk_grad, _wk_grad = mha.W_k.weight.grad, _mha.in_proj_weight.grad.chunk(3)[1]
        wv_grad, _wv_grad = mha.W_v.weight.grad, _mha.in_proj_weight.grad.chunk(3)[2]
        wo_grad, _wo_grad = mha.W_o.weight.grad, _mha.out_proj.weight.grad
        assert torch.allclose(wq_grad, _wq_grad), 'Gradients are not same in W_q'
        assert torch.allclose(wk_grad, _wk_grad), 'Gradients are not same in W_k'
        assert torch.allclose(wv_grad, _wv_grad), 'Gradients are not same in W_v'
        assert torch.allclose(wo_grad, _wo_grad), 'Gradients are not same in W_o'
        
        
    def test_masks(
        self,
        has_attn_mask: bool,
        has_padding_mask: bool
    ) -> None:
        pass
        
        
if __name__ == '__main__':
    pytest.main()
    