"""
Robust Radar I/Q Signal Preprocessing (From Scratch)
(Original 2-Channel Mag+Phase Version - RESTORED)

This version uses a stable, 3-second moving-average filter
to properly suppress respiration artifacts from BOTH channels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

class RadarIQProcessor(nn.Module):
    """
    Convert raw I/Q to magnitude and unwrapped phase
    with a robust, PyTorch-based artifact suppression filter.
    """
    def __init__(self, fs=100): # <-- Note: fs=100 from config
        super().__init__()
        self.fs = fs
        
        # A 3-second window (301 samples at 100Hz) is stable
        # for 0.2-0.3Hz respiration.
        self.trend_kernel_size = (fs * 3) + 1 # 301 samples
        
        self.low_pass_filter = nn.AvgPool1d(
            kernel_size=self.trend_kernel_size,
            stride=1,
            padding=self.trend_kernel_size // 2, # 'same' padding
            count_include_pad=True # Fixes edge artifacts
        )

    def forward(self, radar_i, radar_q):
        """
        Input: radar_i, radar_q [B, L]
        Output: [B, 2, L] (magnitude, phase)
        """
        # --- 1. Calculate Magnitude and Phase ---
        if radar_i.dim() == 3:
            radar_i = radar_i.squeeze(1)
        if radar_q.dim() == 3:
            radar_q = radar_q.squeeze(1)
            
        magnitude = torch.sqrt(radar_i**2 + radar_q**2)
        phase = torch.atan2(radar_q, radar_i)
        
        magnitude = torch.nan_to_num(magnitude, nan=0.0)
        phase = torch.nan_to_num(phase, nan=0.0)

        # --- 2. Unwrap Phase ---
        phase_unwrap = self.unwrap_phase(phase)
        
        # --- 3. High-Pass Filter (Remove Respiration) ---
        phase_unwrap_ch = phase_unwrap.unsqueeze(1)
        magnitude_ch = magnitude.unsqueeze(1)
        
        # Find the slow-moving trend (respiration)
        phase_trend = self.low_pass_filter(phase_unwrap_ch)
        mag_trend = self.low_pass_filter(magnitude_ch)
        
        # Subtract the trend to get the high-pass (heartbeat) signal
        phase_filt = (phase_unwrap_ch - phase_trend).squeeze(1) # [B, L]
        magnitude_filt = (magnitude_ch - mag_trend).squeeze(1) # [B, L]

        # --- 4. Normalize ---
        magnitude_norm = self.normalize(magnitude_filt)
        phase_norm = self.normalize(phase_filt)
        
        magnitude_norm = torch.nan_to_num(magnitude_norm, nan=0.0)
        phase_norm = torch.nan_to_num(phase_norm, nan=0.0)

        return torch.stack([magnitude_norm, phase_norm], dim=1)

    def unwrap_phase(self, phase):
        """
        Remove 2π discontinuities from a phase signal.
        This is a robust PyTorch implementation of np.unwrap.
        Input: [B, L]
        """
        diff = torch.diff(phase, dim=-1)
        
        diff_mod = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        
        # Correct for jumps that land exactly at -pi
        torch.where(
            (diff_mod == -torch.pi) & (diff > 0), 
            torch.tensor(torch.pi, device=phase.device, dtype=phase.dtype), 
            diff_mod,
            out=diff_mod
        )

        offset = diff - diff_mod
        unwrap_offset = torch.cumsum(offset, dim=-1)
        
        phase_unwrap = phase.clone()
        phase_unwrap[..., 1:] = phase[..., 1:] + unwrap_offset
        
        return torch.nan_to_num(phase_unwrap, nan=0.0)

    def normalize(self, x):
        """
        Normalize to zero mean, unit variance.
        Input: [B, L]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-5)

# --- Test the module ---
if __name__ == '__main__':
    # We need to add the parent directory to the path to find 'configs'
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from configs.config import cfg
    except ImportError:
        print("Warning: Could not import config.py. Using default fs=100.")
        class Cfg: fs = 100
        cfg = Cfg()

    processor = RadarIQProcessor(fs=cfg.fs)
    print(f"Testing RadarIQProcessor (2-Channel Mag+Phase Version)...")
    
    B, L = 4, 8000
    i = torch.randn(B, L)
    q = torch.randn(B, L)
    
    # Add a slow 0.2Hz (5-second) respiration wave
    t = torch.linspace(0, L/cfg.fs, L)
    respiration_wave = torch.sin(2 * np.pi * 0.2 * t).unsqueeze(0)
    q = q + (respiration_wave * 10)
    
    print(f"Input shape: {i.shape}")
    
    # Run processor
    output = processor(i, q)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, 2, L)
    print("✓ Output shape is correct (2 channels).")