"""
Multi-Scale Dilated Inception Branch (Regularized)
==================================================
Updates:
- Added Dropout (p=0.3) to every block to prevent severe overfitting.
- This forces the model to learn robust shapes instead of memorizing noise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, dropout_prob=0.3):
        super().__init__()
        # For Conv1d with kernel_size=3 and dilation=d:
        # output_length = (input_length + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
        # To maintain input_length, we need: 2*padding = dilation*(kernel_size-1) = dilation*2
        # Therefore: padding = dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=3, 
            padding=dilation, 
            dilation=dilation, 
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dilation = dilation
        
        # REGULARIZATION: Dropout to stop memorization
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        original_length = x.shape[-1]
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        # Ensure output matches input length
        if x.shape[-1] != original_length:
            if x.shape[-1] > original_length:
                x = x[..., :original_length]
            else:
                # Pad if needed
                pad_amount = original_length - x.shape[-1]
                x = F.pad(x, (0, pad_amount), mode='reflect')
        
        x = self.dropout(x)
        return x

class DilatedInceptionBranch(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.channels = channels
        
        # We want the output to be 4x channels to match the old Wavelet Branch
        self.out_channels = channels * 4
        
        # We divide the output channels among the 4 branches
        branch_channels = self.out_channels // 4 
        
        # 4 Parallel Scales with DROPOUT enabled
        self.branch1 = DilatedInceptionBlock(channels, branch_channels, dilation=1, dropout_prob=0.3)
        self.branch2 = DilatedInceptionBlock(channels, branch_channels, dilation=2, dropout_prob=0.3)
        self.branch3 = DilatedInceptionBlock(channels, branch_channels, dilation=4, dropout_prob=0.3)
        self.branch4 = DilatedInceptionBlock(channels, branch_channels, dilation=8, dropout_prob=0.3)
        
        # Final mixing layer to fuse the scales
        self.mixer = nn.Sequential(
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm1d(self.out_channels),
            nn.GELU(),
            nn.Dropout(0.3) # Extra dropout at the fusion stage
        )

    def forward(self, x):
        """
        Input: [B, C, T]
        Output: [B, 4*C, T]
        """
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # Concatenate all scales
        concat = torch.cat([b1, b2, b3, b4], dim=1)
        
        # Mix
        out = self.mixer(concat)
        
        return out