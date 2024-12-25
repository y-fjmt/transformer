import itertools
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from mha import MultiHeadAttention
import pytest

torch.manual_seed(42)

class TestMultiheadAttention:
    
    
    @pytest.mark.parametrize("batch_size", [2**i for i in range(5)])
    def test_batch_size(self, batch_size: bool) -> None:
        self._test(batch_size=batch_size)
        
    
    @pytest.mark.parametrize("seq_len", [2**i for i in range(5, 11)])
    def test_seq_len(self, seq_len: bool) -> None:
        self._test(seq_len=seq_len)
        
    
    @pytest.mark.parametrize("d_model", [2**i for i in range(3, 11)])
    def test_d_model(self, d_model: bool) -> None:
        self._test(d_model=d_model)
        
        
    @pytest.mark.parametrize("num_heads", [2**i for i in range(3, 7)])
    def test_num_heads(self, num_heads: bool) -> None:
        self._test(num_heads=num_heads)
        
    
    @pytest.mark.parametrize("bias", [True, False])
    def test_bias(self, bias: bool) -> None:
        self._test(bias=bias)
        
        
    @pytest.mark.parametrize("use_attn_mask", [True, False])
    def test_attn_mask(self, use_attn_mask: bool) -> None:
        self._test(use_attn_mask=use_attn_mask)
        
        
    @pytest.mark.parametrize("use_pad_mask", [True, False])
    def test_pad_mask(self, use_pad_mask: bool) -> None:
        self._test(use_pad_mask=use_pad_mask)
        
    
    def _test(
        self,
        batch_size: int = 1,
        seq_len: int = 128,
        d_model: int = 512,
        num_heads: int = 8,
        bias=True,
        use_attn_mask: bool = False,
        use_pad_mask: bool = False
    ) -> None:
        
        _mha = nn.MultiheadAttention(d_model, num_heads, bias=bias, batch_first=True)
        mha = MultiHeadAttention(d_model, num_heads, bias=bias)
        mha.load_weights_from_torch(_mha)
        
        k, q, v = torch.randn(3, batch_size, seq_len, d_model)
        
        attn_mask = None if not use_attn_mask else self._diag_aten_mask(batch_size, num_heads, seq_len)
        pad_mask = None if not use_pad_mask else self._random_pad_mask(batch_size, seq_len)
        
        # forward test
        out, out_w  = mha(k, q, v, attn_mask=attn_mask, key_padding_mask=pad_mask)
        _out, _out_w = _mha(k, q, v, attn_mask=attn_mask, key_padding_mask=pad_mask)
        
        assert torch.allclose(out, _out, atol=1e-6), 'Outputs are not same.'
        
        # backward test
        out.mean().backward()
        _out.mean().backward()
        
        
        _wq_grad, _wk_grad, _wv_grad = _mha.in_proj_weight.grad.chunk(3)
        wq_grad, wk_grad, wv_grad = mha.W_q.weight.grad, mha.W_k.weight.grad, mha.W_v.weight.grad
        wo_grad, _wo_grad = mha.W_o.weight.grad, _mha.out_proj.weight.grad
        
        assert torch.allclose(wq_grad, _wq_grad, atol=1e-8), 'Gradients are not same in W_q'
        assert torch.allclose(wk_grad, _wk_grad, atol=1e-8), 'Gradients are not same in W_k'
        assert torch.allclose(wv_grad, _wv_grad, atol=1e-8), 'Gradients are not same in W_v'
        assert torch.allclose(wo_grad, _wo_grad, atol=1e-8), 'Gradients are not same in W_o'
        
        if bias:
            _bq_grad, _bk_grad, _bv_grad = _mha.in_proj_bias.grad.chunk(3)
            bq_grad, bk_grad, bv_grad = mha.W_q.bias.grad, mha.W_k.bias.grad, mha.W_v.bias.grad
            bo_grad, _bo_grad = mha.W_o.bias.grad, _mha.out_proj.bias.grad
            assert torch.allclose(bq_grad, _bq_grad, atol=1e-8), 'Gradients are not same in B_q'
            assert torch.allclose(bk_grad, _bk_grad, atol=1e-8), 'Gradients are not same in B_k'
            assert torch.allclose(bv_grad, _bv_grad, atol=1e-8), 'Gradients are not same in B_v'
            assert torch.allclose(bo_grad, _bo_grad, atol=1e-8), 'Gradients are not same in B_o'
    
    
    @staticmethod
    def _random_pad_mask(
            N: int, 
            seq_len: int
        ) -> torch.Tensor:
        
        arr = torch.arange(0, seq_len).expand(N, seq_len)
        n_zeros = torch.randint(0, seq_len-1, (N, )).unsqueeze(1)
        return torch.where(arr >= n_zeros, float('-inf'), 0.0)
    
    
    @staticmethod
    def _diag_aten_mask(
        N: int,
        num_heads: int,
        seq_len: int,
    ) -> torch.Tensor:
        return (
            nn.Transformer.generate_square_subsequent_mask(seq_len)
            .unsqueeze(0)
            .expand(N*num_heads, -1, -1))

        
        
if __name__ == '__main__':
    pytest.main()