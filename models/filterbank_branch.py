"""
Learned Filterbank Branch (Fixed)
- Removed internal downsampling to match Inception Branch size.
- Fixed AMP warning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LearnedFilterBank(nn.Module):
    def __init__(self, channels=32, n_filters=8, fs=128):
        super().__init__()
        self.channels = channels
        self.n_filters = n_filters
        self.fs = fs
        
        # Learnable parameters: center frequency and bandwidth
        centers_init = torch.linspace(5, 50, n_filters)
        self.log_centers = nn.Parameter(torch.log(centers_init))
        self.log_bandwidth = nn.Parameter(torch.ones(n_filters) * np.log(5.0))
        
    def forward(self, x):
        """
        Input: [B, C, T]
        Output: [B, C*n_filters, T] (Full Resolution)
        """
        B, C, T = x.shape
        
        # --- FIX: Updated AMP syntax ---
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float() 
            
            # 1. FFT
            X = torch.fft.rfft(x, dim=-1)
            freqs = torch.fft.rfftfreq(T, d=1/self.fs).to(x.device)
            
            # 2. Prepare Filters
            centers = torch.exp(self.log_centers.float())
            bandwidths = torch.exp(self.log_bandwidth.float())
            
            centers = torch.clamp(centers, 1.0, self.fs/2 - 1)
            bandwidths = torch.clamp(bandwidths, 1.0, 20.0)
            
            # 3. Apply Filters
            outputs = []
            for i in range(self.n_filters):
                H = self.gaussian_bandpass(freqs, centers[i], bandwidths[i])
                X_filt = X * H.view(1, 1, -1)
                x_filt = torch.fft.irfft(X_filt, n=T, dim=-1)
                outputs.append(x_filt)
            
            output = torch.cat(outputs, dim=1)
        
        # --- FIX: REMOVED avg_pool1d ---
        # We return the full length T to match the Time branch.
        # Downsampling is handled by the main model's 'downsamples' layer.
        
        return output
    
    def gaussian_bandpass(self, freqs, center, bandwidth):
        sigma = bandwidth / (2 * np.sqrt(2 * np.log(2))) + 1e-8
        H = torch.exp(-0.5 * ((freqs - center) / (sigma + 1e-8))**2)
        return H