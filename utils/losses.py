"""
Multi-Channel Losses (Fixed for 2-Lead ECG)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask):
        # Align shapes if needed
        if pred.shape[-1] != target.shape[-1]:
            min_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_len]
            target = target[..., :min_len]
            if mask.dim() == 2:
                mask = mask[..., :min_len]
        
        if pred.dim() == 3 and mask.dim() == 2:
            mask = mask.unsqueeze(1) # Broadcast mask to channels
        
        abs_error = torch.abs(pred - target)
        masked_error = abs_error * mask
        return masked_error.sum() / (mask.sum() * pred.shape[1] + 1e-8)

class PeakLoss(nn.Module):
    """
    Focus on R-peaks using Binary Cross Entropy on the auxiliary peak head.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, target_ecg, mask):
        # Align shapes if needed
        if pred_logits.shape[-1] != target_ecg.shape[-1]:
            min_len = min(pred_logits.shape[-1], target_ecg.shape[-1])
            pred_logits = pred_logits[..., :min_len]
            target_ecg = target_ecg[..., :min_len]
            if mask.dim() == 2:
                mask = mask[..., :min_len]
        
        # Generate soft targets on the fly based on amplitude
        with torch.no_grad():
            batch_max = target_ecg.max(dim=-1, keepdim=True)[0]
            peak_targets = (target_ecg > (0.7 * batch_max)).float()
            
        if mask.dim() == 2: mask = mask.unsqueeze(1)
        
        loss = self.bce(pred_logits, peak_targets) * mask
        return loss.sum() / (mask.sum() + 1e-8)

class TemporalCorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        # Align shapes if needed
        if pred.shape[-1] != target.shape[-1]:
            min_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_len]
            target = target[..., :min_len]
            if mask.dim() == 2:
                mask = mask[..., :min_len]
        
        # Handle [B, C, T] inputs
        if mask.dim() == 2: mask = mask.unsqueeze(1)
        
        mean_p = pred.mean(dim=-1, keepdim=True)
        mean_t = target.mean(dim=-1, keepdim=True)
        
        p_centered = (pred - mean_p) * mask
        t_centered = (target - mean_t) * mask
        
        numerator = (p_centered * t_centered).sum(dim=-1)
        
        std_p = torch.sqrt((p_centered ** 2).sum(dim=-1)).clamp(min=1e-6)
        std_t = torch.sqrt((t_centered ** 2).sum(dim=-1)).clamp(min=1e-6)
        
        pcc = numerator / (std_p * std_t)
        
        # Average over batch AND channels (1.0 - mean_pcc)
        return 1.0 - torch.nan_to_num(pcc, nan=0.0).mean()

class SpectralCorrelationLoss(nn.Module):
    def __init__(self, n_fft=256, hop=64, win=256, fs=128.0, max_freq=40.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.register_buffer('window', torch.hann_window(win))

    def forward(self, pred, target, mask):
        # Align shapes if needed
        if pred.shape[-1] != target.shape[-1]:
            min_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_len]
            target = target[..., :min_len]
        
        # Reshape to combine Batch and Channel for STFT
        # [B, C, T] -> [B*C, T]
        B, C, T = pred.shape
        pred_flat = pred.view(B*C, T)
        target_flat = target.view(B*C, T)
        
        p_stft = torch.stft(pred_flat, self.n_fft, self.hop, self.win, self.window, return_complex=True)
        t_stft = torch.stft(target_flat, self.n_fft, self.hop, self.win, self.window, return_complex=True)
        
        p_mag = torch.abs(p_stft) + 1e-8
        t_mag = torch.abs(t_stft) + 1e-8
        
        p_norm = F.normalize(p_mag, p=2, dim=1)
        t_norm = F.normalize(t_mag, p=2, dim=1)
        
        spectral_sim = (p_norm * t_norm).sum(dim=1)
        return 1.0 - spectral_sim.mean()

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        # Align shapes if needed
        if mask.dim() == 2 and mask.shape[-1] != pred.shape[-1]:
            min_len = min(mask.shape[-1], pred.shape[-1])
            pred = pred[..., :min_len]
            mask = mask[..., :min_len]
        
        # pred: [B, C, T]
        diff = torch.abs(pred[..., 1:] - pred[..., :-1])
        if mask.dim() == 2: mask = mask.unsqueeze(1)
        
        mask_sliced = mask[..., 1:]
        loss = (diff * mask_sliced).sum() / (mask_sliced.sum() * pred.shape[1] + 1e-8)
        return loss

class NegativePenaltyLoss(nn.Module):
    def __init__(self, penalty_weight=5.0):
        super().__init__()
        self.penalty = penalty_weight
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, pred, target, mask):
        # Align shapes if needed
        if pred.shape[-1] != target.shape[-1]:
            min_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_len]
            target = target[..., :min_len]
            if mask.dim() == 2:
                mask = mask[..., :min_len]
        
        pixel_loss = self.l1(pred, target)
        
        weights = torch.ones_like(target)
        weights[target < 0] = self.penalty
        
        if mask.dim() == 2: mask = mask.unsqueeze(1)
        
        weighted_loss = pixel_loss * weights * mask
        return weighted_loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)

class DirectionalSlopeLoss(nn.Module):
    def __init__(self, negative_slope_weight=4.0):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.neg_weight = negative_slope_weight

    def forward(self, pred, target, mask):
        # Align shapes if needed
        if pred.shape[-1] != target.shape[-1]:
            min_len = min(pred.shape[-1], target.shape[-1])
            pred = pred[..., :min_len]
            target = target[..., :min_len]
            if mask.dim() == 2:
                mask = mask[..., :min_len]
        
        pred_slope = pred[..., 1:] - pred[..., :-1]
        target_slope = target[..., 1:] - target[..., :-1]
        
        if mask.dim() == 2: mask = mask.unsqueeze(1)
        mask_sliced = mask[..., 1:]
        
        loss = self.l1(pred_slope, target_slope)
        
        weight_map = torch.ones_like(target_slope)
        weight_map[target_slope < 0] = self.neg_weight
        
        weighted_loss = loss * weight_map * mask_sliced
        return weighted_loss.sum() / (mask_sliced.sum() * pred.shape[1] + 1e-8)

class CompleteLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = MaskedL1Loss()
        self.peak = PeakLoss()
        self.temp_corr = TemporalCorrelationLoss()
        self.spec_corr = SpectralCorrelationLoss()
        self.slope = DirectionalSlopeLoss()
        self.neg_loss = NegativePenaltyLoss()
        self.tv = TotalVariationLoss()
        
        # Weights
        self.w_l1 = config.lambda_l1
        self.w_peak = config.lambda_peak
        self.w_temp = config.lambda_temp_corr
        self.w_spec = config.lambda_spec_corr
        self.w_slope = config.lambda_slope
        self.w_neg = config.lambda_neg
        
        self.w_bp_l1 = config.lambda_bp_l1
        self.w_bp_corr = config.lambda_bp_corr
        self.w_icg_l1 = config.lambda_icg_l1
        self.w_icg_corr = config.lambda_icg_corr
        self.w_dicg_corr = config.lambda_dicg_corr
        self.w_strain_l1 = config.lambda_strain_l1
        self.w_resp_l1 = config.lambda_resp_l1
        self.w_tv = config.lambda_tv

    def forward(self, outputs, targets, mask, task_flags=None):
        """flags: [B, 3] -> [BP, Strain, Resp]"""
        
        # --- 1. ECG ---
        raw_ecg_l1 = self.l1(outputs['ecg'], targets['ecg'], mask)
        raw_ecg_temp = self.temp_corr(outputs['ecg'], targets['ecg'], mask)
        raw_ecg_spec = self.spec_corr(outputs['ecg'], targets['ecg'], mask)
        raw_ecg_slope = self.slope(outputs['ecg'], targets['ecg'], mask)
        raw_ecg_neg = self.neg_loss(outputs['ecg'], targets['ecg'], mask)
        
        tgt_peak = targets['ecg'][:, 1:2] 
        raw_ecg_peak = self.peak(outputs['peak_logits'], tgt_peak, mask)

        l_ecg = (raw_ecg_l1 * self.w_l1) + \
                (raw_ecg_temp * self.w_temp) + \
                (raw_ecg_spec * self.w_spec) + \
                (raw_ecg_slope * self.w_slope) + \
                (raw_ecg_neg * self.w_neg) + \
                (raw_ecg_peak * self.w_peak)
        
        # --- 2. ICG & Flow ---
        raw_icg_l1 = self.l1(outputs['icg'], targets['icg'], mask)
        raw_icg_corr = self.temp_corr(outputs['icg'], targets['icg'], mask)
        l_icg = (raw_icg_l1 * self.w_icg_l1) + (raw_icg_corr * self.w_icg_corr)
        
        raw_dicg_corr = self.temp_corr(outputs['dicg'], targets['dicg'], mask)
        l_dicg = raw_dicg_corr * self.w_dicg_corr
        
        # --- 3. Conditional Losses ---
        # BP
        raw_bp_l1 = self.l1(outputs['bp'], targets['bp'], mask)
        raw_bp_corr = self.temp_corr(outputs['bp'], targets['bp'], mask)
        
        l_bp_combined = (raw_bp_l1 * self.w_bp_l1) + (raw_bp_corr * self.w_bp_corr)
        if task_flags is not None:
            l_bp = (l_bp_combined * task_flags[:, 0].mean())
        else:
            l_bp = l_bp_combined
        
        # Strain (only if in outputs)
        l_strain = torch.tensor(0.0, device=l_ecg.device, dtype=l_ecg.dtype)
        if 'strain' in outputs and 'strain' in targets:
            raw_strain_l1 = self.l1(outputs['strain'], targets['strain'], mask)
            if task_flags is not None:
                l_strain = (raw_strain_l1 * self.w_strain_l1) * task_flags[:, 1].mean()
            else:
                l_strain = raw_strain_l1 * self.w_strain_l1
        
        # Resp (only if in outputs)
        l_resp = torch.tensor(0.0, device=l_ecg.device, dtype=l_ecg.dtype)
        if 'resp' in outputs and 'resp' in targets:
            raw_resp_l1 = self.l1(outputs['resp'], targets['resp'], mask)
            raw_resp_tv = self.tv(outputs['resp'], mask)
            l_resp_combined = (raw_resp_l1 * self.w_resp_l1) + (raw_resp_tv * self.w_tv)
            if task_flags is not None:
                l_resp = l_resp_combined * task_flags[:, 2].mean()
            else:
                l_resp = l_resp_combined

        total = l_ecg + l_icg + l_dicg + l_bp + l_strain + l_resp
        
        # --- RETURN DETAILED KEYS (The Fix) ---
        return {
            'total': total,
            # Keys required by train.py:
            'ecg_temp_score': 100 * (1 - raw_ecg_temp),
            'ecg_spec_score': 100 * (1 - raw_ecg_spec),
            'ecg_l1_score':   100 * (1 - raw_ecg_l1), 
            'bp_corr_score':  100 * (1 - raw_bp_corr),
            'icg_corr_score': 100 * (1 - raw_icg_corr),
            'flow_corr_score':100 * (1 - raw_dicg_corr),
        }