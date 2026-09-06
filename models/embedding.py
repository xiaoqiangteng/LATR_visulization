import torch
import torch.nn as nn
import math


class TimeStepEmbedding(nn.Module):
    # learned from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py
    def __init__(self, dim=256, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.linear = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim // 2),
        )

        self.out_dim = dim // 2

    def _compute_freqs(self, half):
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        return freqs

    def forward(self, timesteps):
        half = self.dim // 2
        freqs = self._compute_freqs(half).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        output = self.linear(embedding)
        return output


class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=6, append_input=True):
        super().__init__()
        self.n_harmonic_functions = n_harmonic_functions
        self.append_input = append_input

        self.freq_bands = 2 ** torch.arange(n_harmonic_functions).float() * math.pi  # [π, 2π, 4π, ..., 2^n * π]

    def forward(self, x):
        """
        x: Tensor of shape [..., D]
        Returns: Harmonic embedding of shape [..., D * (2 * n_harmonic_functions [+1])]
        """
        x = x.unsqueeze(-1)  # [..., D, 1]
        embed = x * self.freq_bands.to(x.device)  # [..., D, H]
        sin = torch.sin(embed)  # [..., D, H]
        cos = torch.cos(embed)  # [..., D, H]
        harmonic = torch.cat([sin, cos], dim=-1)  # [..., D, 2H]
        harmonic = harmonic.view(*x.shape[:-2], -1)  # flatten last 2 dims

        if self.append_input:
            harmonic = torch.cat([x.squeeze(-1), harmonic], dim=-1)
        return harmonic

    def get_output_dim(self, input_dim: int) -> int:
        """
        Computes the output dimension after harmonic embedding.
        """
        dim = input_dim * self.n_harmonic_functions * 2  # sin & cos
        if self.append_input:
            dim += input_dim
        return dim


class PoseEmbedding(nn.Module):
    def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
        super().__init__()

        self._emb_pose = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=append_input
        )

        self.out_dim = self._emb_pose.get_output_dim(target_dim)

    def forward(self, pose_encoding):
        e_pose_encoding = self._emb_pose(pose_encoding)
        return e_pose_encoding
