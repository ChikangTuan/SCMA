import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PatchPadMixin:
    patch_size = 8

    def _pad_to_patch_size(self, x):
        h, w = x.shape[-2:]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x, h, w
        mode = "reflect" if pad_h < h and pad_w < w else "replicate"
        return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), h, w


class DFFN(nn.Module, PatchPadMixin):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.fft = nn.Parameter(
            torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = x.to(self.project_in.weight.dtype)
        x, orig_h, orig_w = self._pad_to_patch_size(x)
        x = self.project_in(x)
        x_patch = rearrange(
            x,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(
            x_patch,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x = x.to(self.dwconv.weight.dtype)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = x.to(self.project_out.weight.dtype)
        x = self.project_out(x)
        return x[..., :orig_h, :orig_w]


class FSAS(nn.Module, PatchPadMixin):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(
            dim * 6,
            dim * 6,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 6,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type="WithBias")

    def forward(self, x):
        x = x.to(self.to_hidden.weight.dtype)
        x, orig_h, orig_w = self._pad_to_patch_size(x)
        hidden = self.to_hidden(x)
        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        q_patch = rearrange(
            q,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        k_patch = rearrange(
            k,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(
            out,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        out = out.to(self.norm.body.weight.dtype)
        out = self.norm(out)
        output = v * out.to(v.dtype)
        output = output.to(self.project_out.weight.dtype)
        output = self.project_out(output)
        return output[..., :orig_h, :orig_w]


class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias"):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FSAS(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
