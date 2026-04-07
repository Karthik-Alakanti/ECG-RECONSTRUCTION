"""
CAST-ECG (Final Phase 3 Architecture)
=====================================
A Physics-Aware, Morphology-Constrained Deep Learning Model
for Non-Contact ECG Reconstruction.

Key Components:
1. Dual-Branch Feature Extraction:
   - Time Domain: Dilated Inception Branch (Learnable Multi-Scale Edges)
   - Freq Domain: Learned Filterbank Branch (Spectral Rhythms)
2. SpatioTemporal Router: Context-aware dynamic gating.
3. Learnable Phase Shift: FFT-based alignment layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- DEPENDENCIES ---
# Ensure these files exist in your 'models/' folder
from .incept import DilatedInceptionBranch
from .filterbank_branch import LearnedFilterBank

class LearnablePhaseShift(nn.Module):
    def __init__(self, max_shift_samples=20):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1))
        self.max_shift = max_shift_samples
    def forward(self, x):
        B, C, T = x.shape
        x_fft = torch.fft.rfft(x.float(), dim=-1)
        f = torch.arange(x_fft.shape[-1], device=x.device)
        tau = torch.tanh(self.shift) * self.max_shift
        phase_shift = -2 * np.pi * f * tau / T
        rotation = torch.complex(torch.cos(phase_shift), torch.sin(phase_shift))
        x_shifted_fft = x_fft * rotation.view(1, 1, -1)
        return torch.fft.irfft(x_shifted_fft, n=T, dim=-1).to(x.dtype)

class ResidualConvBlock(nn.Module):
    """
    Standard Residual Block with Dilation for stable feature refinement.
    """
    def __init__(self, channels, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, 
                               padding=1, bias=False) # No dilation
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act(x + res)

class SpatioTemporalRouter(nn.Module):
    """
    The "Smart Mixer" (Explainable AI Component).
    Generates a time-varying gate map (0.0 to 1.0) to switch between branches.
    """
    def __init__(self, channels):
        super().__init__()
        # Context-aware gating network
        self.context_net = nn.Sequential(
            # Large kernel (7) sees local context (Spike vs Flat?)
            nn.Conv1d(channels * 2, channels, kernel_size=7, padding=3, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            # Pointwise projection to single Gate channel
            nn.Conv1d(channels, 1, kernel_size=7, padding=3), 
            nn.Sigmoid() # Force output 0.0 (Filterbank) to 1.0 (Inception)
        )
        self.out_proj = nn.Conv1d(channels, channels, 1)

    def forward(self, feat_time, feat_freq):
        # 1. Concatenate Features
        combined = torch.cat([feat_time, feat_freq], dim=1)
        
        # 2. Predict Gate Map [B, 1, T]
        gate = self.context_net(combined)
        
        # 3. Soft Mixing
        fused = (feat_time * gate) + (feat_freq * (1 - gate))
        
        # 4. Final Projection
        out = self.out_proj(fused)
        
        return out, gate

class SimplifiedCASTECG_Paper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_scales = config.n_scales
        
        self.in_conv = nn.Conv1d(config.in_channels, config.base_channels, kernel_size=31, padding=15)
        self.branch_time = DilatedInceptionBranch(channels=config.base_channels)
        self.branch_freq = LearnedFilterBank(channels=config.base_channels, n_filters=config.n_filters, fs=config.fs)
        
        # U-Net Structure
        self.routers = nn.ModuleList()
        self.refiners = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.time_projs = nn.ModuleList()
        self.freq_projs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        curr_dim = config.base_channels
        
        for i in range(self.n_scales):
            self.time_projs.append(nn.Conv1d(4 * curr_dim, curr_dim, 1))
            self.freq_projs.append(nn.Conv1d(config.n_filters * curr_dim, curr_dim, 1))
            self.routers.append(SpatioTemporalRouter(curr_dim))
            self.refiners.append(nn.Sequential(ResidualConvBlock(curr_dim, dilation=1), ResidualConvBlock(curr_dim, dilation=2)))
            if i < self.n_scales - 1:
                self.downsamples.append(nn.Conv1d(curr_dim, curr_dim, kernel_size=3, stride=2, padding=1))
        
        for i in range(self.n_scales - 1):
            self.upsamples.append(nn.ConvTranspose1d(curr_dim, curr_dim, kernel_size=4, stride=2, padding=1))
            self.decoders.append(ResidualConvBlock(curr_dim))

        # --- 5 HEADS (Removed Strain and Respiration) ---
        dim = curr_dim
        
        # 1. ECG (2-Leads: I, II only)
        self.head_ecg = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.GELU(), nn.Conv1d(dim, 2, 3, padding=1))
        self.phase_compensator = LearnablePhaseShift(max_shift_samples=20)
        
        # 2. BP
        self.head_bp = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.GELU(), nn.Conv1d(dim, 1, 3, padding=1))
        # 3. ICG
        self.head_icg = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.GELU(), nn.Conv1d(dim, 1, 3, padding=1))
        # 4. Flow
        self.head_dicg = nn.Sequential(nn.Conv1d(dim, dim, 1), nn.GELU(), nn.Conv1d(dim, 1, 3, padding=1))
        # 5. Peak
        self.head_peak = nn.Sequential(nn.Conv1d(dim, dim//2, 1), nn.GELU(), nn.Conv1d(dim//2, 1, 3, padding=1))

    def forward(self, x, epoch=None):
        x = self.in_conv(x)
        skips, gates = [], []
        
        for i in range(self.n_scales):
            t = self.time_projs[i](self.branch_time(x))
            f = self.freq_projs[i](self.branch_freq(x))
            if t.shape[-1] != x.shape[-1]: t = F.interpolate(t, size=x.shape[-1])
            if f.shape[-1] != x.shape[-1]: f = F.interpolate(f, size=x.shape[-1])
            
            fused, gate = self.routers[i](t, f)
            gates.append(gate)
            fused = self.refiners[i](fused)
            skips.append(fused)
            if i < self.n_scales - 1: x = self.downsamples[i](fused)
                
        y = skips[-1]
        for i in reversed(range(self.n_scales - 1)):
            y = self.upsamples[i](y)
            skip = skips[i]
            if y.shape[-1] != skip.shape[-1]: y = y[..., :skip.shape[-1]]
            y = self.decoders[i](y + skip)
            
        if y.shape[-1] != self.config.max_input_length:
             y = F.interpolate(y, size=self.config.max_input_length)

        return {
            'ecg': self.phase_compensator(self.head_ecg(y)),
            'bp': self.head_bp(y),
            'icg': self.head_icg(y),
            'dicg': self.head_dicg(y),
            'peak_logits': self.head_peak(y)
        }