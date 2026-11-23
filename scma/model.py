from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fft_transformer import TransformerBlock


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period)
        * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels, eps=1e-5, affine=True)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: th.Tensor) -> th.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, hidden_states: th.Tensor, temb: th.Tensor) -> th.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb.to(dtype=hidden_states.dtype)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return hidden_states + residual


class AttentionBlock2D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = _group_norm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(0.0)])

    def forward(self, hidden_states: th.Tensor) -> th.Tensor:
        residual = hidden_states
        bsz, channels, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, channels, height * width).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        head_dim = channels // self.num_heads
        query = query.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(bsz, -1, self.num_heads, head_dim).transpose(1, 2)

        scale = head_dim**-0.5
        attn = th.matmul(query, key.transpose(-1, -2)) * scale
        attn = th.softmax(attn.float(), dim=-1).to(dtype=value.dtype)
        hidden_states = th.matmul(attn, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(bsz, height * width, channels)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(bsz, channels, height, width)
        return hidden_states + residual


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, hidden_states: th.Tensor) -> th.Tensor:
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: th.Tensor) -> th.Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        layers_per_block: int,
        dropout: float,
        add_downsample: bool,
        add_attention: bool = False,
        add_fft_transformer: bool = False,
        num_add_res: int = 1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if add_attention else None
        self.fft_transformers = nn.ModuleList() if add_fft_transformer else None
        for i in range(layers_per_block):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(res_in, out_channels, temb_channels, dropout))
            if add_attention:
                self.attentions.append(AttentionBlock2D(out_channels))
            if add_fft_transformer:
                self.fft_transformers.append(
                    nn.Sequential(*[TransformerBlock(out_channels) for _ in range(num_add_res)])
                )
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else None

    def forward(self, hidden_states: th.Tensor, temb: th.Tensor):
        output_states = []
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states, temb)
            if self.fft_transformers is not None:
                hidden_states = self.fft_transformers[i](hidden_states)
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states)
            output_states.append(hidden_states)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)
        return hidden_states, tuple(output_states)


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        layers_per_block: int,
        dropout: float,
        add_upsample: bool,
        add_attention: bool = False,
        add_fft_transformer: bool = False,
        num_add_res: int = 1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if add_attention else None
        self.fft_transformers = nn.ModuleList() if add_fft_transformer else None
        for i in range(layers_per_block + 1):
            res_skip_channels = in_channels if i == layers_per_block else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    resnet_in_channels + res_skip_channels,
                    out_channels,
                    temb_channels,
                    dropout,
                )
            )
            if add_attention:
                self.attentions.append(AttentionBlock2D(out_channels))
            if add_fft_transformer:
                self.fft_transformers.append(
                    nn.Sequential(*[TransformerBlock(out_channels) for _ in range(num_add_res)])
                )
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None

    def forward(
        self,
        hidden_states: th.Tensor,
        res_hidden_states_tuple: Tuple[th.Tensor, ...],
        temb: th.Tensor,
    ):
        for i, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = th.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            if self.fft_transformers is not None:
                hidden_states = self.fft_transformers[i](hidden_states)
            if self.attentions is not None:
                hidden_states = self.attentions[i](hidden_states)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, temb_channels: int, dropout: float):
        super().__init__()
        self.resnet_1 = ResnetBlock2D(channels, channels, temb_channels, dropout)
        self.attention = AttentionBlock2D(channels)
        self.resnet_2 = ResnetBlock2D(channels, channels, temb_channels, dropout)

    def forward(self, hidden_states: th.Tensor, temb: th.Tensor) -> th.Tensor:
        hidden_states = self.resnet_1(hidden_states, temb)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.resnet_2(hidden_states, temb)
        return hidden_states


class UNet2DModel(nn.Module):
    def __init__(
        self,
        sample_size: int = 416,
        in_channels: int = 2,
        out_channels: int = 1,
        layers_per_block: int = 2,
        block_out_channels: Sequence[int] = (64, 64, 128, 128, 256, 256),
        down_block_types: Sequence[str] = (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types: Sequence[str] = (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        dropout: float = 0.0,
        use_fft_transformer: bool = True,
        num_add_res: int = 1,
    ):
        super().__init__()
        if len(block_out_channels) != len(down_block_types):
            raise ValueError("block_out_channels and down_block_types must have the same length")
        if len(block_out_channels) != len(up_block_types):
            raise ValueError("block_out_channels and up_block_types must have the same length")

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.block_out_channels = tuple(block_out_channels)
        self.down_block_types = tuple(down_block_types)
        self.up_block_types = tuple(up_block_types)
        self.use_fft_transformer = bool(use_fft_transformer)
        self.num_add_res = int(num_add_res)

        time_embed_dim = block_out_channels[0] * 4
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim)

        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            add_downsample = i != len(block_out_channels) - 1
            add_attention = down_block_type == "AttnDownBlock2D"
            add_fft_transformer = self.use_fft_transformer and i >= 2
            if down_block_type not in ("DownBlock2D", "AttnDownBlock2D"):
                raise ValueError(f"unsupported down block: {down_block_type}")
            self.down_blocks.append(
                DownBlock2D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    layers_per_block=layers_per_block,
                    dropout=dropout,
                    add_downsample=add_downsample,
                    add_attention=add_attention,
                    add_fft_transformer=add_fft_transformer,
                    num_add_res=self.num_add_res,
                )
            )

        self.mid_block = UNetMidBlock2D(block_out_channels[-1], time_embed_dim, dropout)

        self.up_blocks = nn.ModuleList()
        reversed_block_out_channels = list(reversed(block_out_channels))
        prev_output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            add_upsample = i != len(block_out_channels) - 1
            add_attention = up_block_type == "AttnUpBlock2D"
            original_level = len(block_out_channels) - 1 - i
            add_fft_transformer = self.use_fft_transformer and original_level >= 2
            if up_block_type not in ("UpBlock2D", "AttnUpBlock2D"):
                raise ValueError(f"unsupported up block: {up_block_type}")
            self.up_blocks.append(
                UpBlock2D(
                    in_channels=input_channel,
                    prev_output_channel=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    layers_per_block=layers_per_block,
                    dropout=dropout,
                    add_upsample=add_upsample,
                    add_attention=add_attention,
                    add_fft_transformer=add_fft_transformer,
                    num_add_res=self.num_add_res,
                )
            )
            prev_output_channel = output_channel

        self.conv_norm_out = _group_norm(block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample: th.Tensor, timestep: th.Tensor) -> th.Tensor:
        if not th.is_tensor(timestep):
            timestep = th.tensor([timestep], dtype=th.long, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(device=sample.device)
        if timestep.shape[0] == 1 and sample.shape[0] > 1:
            timestep = timestep.expand(sample.shape[0])

        temb = timestep_embedding(timestep, self.block_out_channels[0])
        temb = temb.to(dtype=sample.dtype)
        temb = self.time_embedding(temb)

        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, res_samples = down_block(sample, temb)
            down_block_res_samples = down_block_res_samples + res_samples

        sample = self.mid_block(sample, temb)

        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]
            sample = up_block(sample, res_samples, temb)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class VelocityUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DModel()

    def forward(self, state: th.Tensor, condition: th.Tensor, t: th.Tensor) -> th.Tensor:
        if state.ndim != 4 or state.shape[1] != 1 or state.shape != condition.shape:
            raise ValueError("state and LI condition must have the same [batch, 1, height, width] shape")
        if any(side % 32 != 0 for side in state.shape[-2:]):
            raise ValueError("image dimensions must be divisible by 32 for the six U-Net levels")
        if t.shape != (state.shape[0],):
            raise ValueError("t must contain one continuous time per image")
        inputs = th.cat([state, condition], dim=1)
        return self.unet(inputs, 1000.0 * t)
