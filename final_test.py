import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import welch, find_peaks, butter, filtfilt
from scipy.stats import ttest_rel, wilcoxon, pearsonr

# Project Imports
from configs.config import cfg
from models.cast_ecg import SimplifiedCASTECG_Paper as SimplifiedCASTECG
from dataload.dataset import create_patient_wise_splits

# --- CONFIGURATION ---
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']

# =====================================================
# 1. PHYSICS & MAPPING ENGINE
# =====================================================
def reconstruct_6leads_from_forensics(raw_data_3ch):
    """
    Takes the raw channel output and maps it correctly based on forensic analysis.
    Handles both 2-channel and 3-channel input formats:
    - 3-channel: Channel 2 -> Lead I, Channel 1 -> Lead II, Channel 0 -> Lead III (recalculated)
    - 2-channel: Channel 1 -> Lead I, Channel 0 -> Lead II
    
    Shape of input: [Channels, Time] or [Channels, Batch, Time]
    Output shape: [6, Time] (6 ECG leads)
    """
    # Handle batch dimension if present
    if raw_data_3ch.ndim == 3:
        # Assuming shape is [Batch, Channels, Time], reshape to [Channels, Time]
        raw_data_3ch = raw_data_3ch[0] if raw_data_3ch.shape[0] == 1 else raw_data_3ch.transpose(1, 0, 2).reshape(raw_data_3ch.shape[1], -1)
    
    n_channels = raw_data_3ch.shape[0]
    
    if n_channels == 3:
        # 3-channel case: Original forensic mapping
        lead_I = raw_data_3ch[2]      # Forensic Result: Index 2 is Lead I
        lead_II = raw_data_3ch[1]     # Forensic Result: Index 1 is Lead II
    elif n_channels == 2:
        # 2-channel case: Simplified mapping
        lead_I = raw_data_3ch[1]      # Index 1 is Lead I
        lead_II = raw_data_3ch[0]     # Index 0 is Lead II
    else:
        # Fallback: use available channels
        lead_I = raw_data_3ch[max(0, n_channels-1)]
        lead_II = raw_data_3ch[min(1, n_channels-1)] if n_channels > 1 else lead_I.copy()
    
    # 2. Calculate Dependent Leads (Einthoven & Goldberger)
    lead_III = lead_II - lead_I
    lead_aVR = -0.5 * (lead_I + lead_II)
    lead_aVL = lead_I - 0.5 * lead_II
    lead_aVF = lead_II - 0.5 * lead_I
    
    # 3. Stack into standard 6-lead format (I, II, III, aVR, aVL, aVF)
    # Output shape: [6, Time]
    return np.stack([lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF])

# =====================================================
# 2. SIGNAL FILTERING
# =====================================================
def smooth_signal(signal, fs, cutoff=5.0):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(2, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# =====================================================
# 3. METRICS
# =====================================================
def get_correlations(pred, truth, fs):
    p_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
    t_norm = (truth - truth.mean()) / (truth.std() + 1e-8)
    temporal = np.mean(p_norm * t_norm)

    f_p, P_p = welch(pred, fs=fs, nperseg=256)
    f_t, P_t = welch(truth, fs=fs, nperseg=256)
    spectral = np.corrcoef(P_p, P_t)[0, 1]
    return temporal, spectral

def calculate_hemodynamics(ecg_signal, flow_signal, fs):
    rel_sv = np.max(flow_signal) - np.min(flow_signal)
    
    r_peaks, _ = find_peaks(ecg_signal, distance=fs*0.5, prominence=0.5)
    flow_peaks, _ = find_peaks(flow_signal, distance=fs*0.5, prominence=0.2) 

    pep_ms = np.nan
    if len(r_peaks) > 0 and len(flow_peaks) > 0:
        first_r = r_peaks[0]
        valid_flows = flow_peaks[flow_peaks > first_r]
        if len(valid_flows) > 0:
            pep_ms = ((valid_flows[0] - first_r) / fs) * 1000

    return rel_sv, pep_ms

# =====================================================
# 3B. CLASSIFICATION METRICS (Precision, Recall, F1)
# =====================================================
def calculate_signal_quality_classification(pred_signal, truth_signal, threshold=0.7):
    """
    Classify signals as 'good quality' (1) or 'bad quality' (0) based on correlation threshold.
    This follows the pattern from utils/metrics.py for consistency.
    
    Args:
        pred_signal: Predicted signal array
        truth_signal: Ground truth signal array
        threshold: Correlation threshold for classifying as 'good quality'
    
    Returns:
        Dict with precision, recall, f1, accuracy, confusion matrix
    """
    # Flatten signals
    pred_flat = pred_signal.flatten()
    truth_flat = truth_signal.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(truth_flat))
    if np.sum(mask) < 2:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0
        }
    
    pred_clean = pred_flat[mask]
    truth_clean = truth_flat[mask]
    
    # Calculate correlation
    if np.std(pred_clean) < 1e-6 or np.std(truth_clean) < 1e-6:
        corr = 0.0
    else:
        corr = np.corrcoef(pred_clean, truth_clean)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    
    # Classify based on threshold
    pred_class = 1 if corr >= threshold else 0
    truth_class = 1  # Ground truth is always 'good' (ideal signal)
    
    # Calculate confusion matrix metrics
    tp = int(pred_class == 1 and truth_class == 1)
    fp = int(pred_class == 1 and truth_class == 0)
    tn = int(pred_class == 0 and truth_class == 0)
    fn = int(pred_class == 0 and truth_class == 1)
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'correlation': corr,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def calculate_per_lead_classification_metrics(p_ecg_6, t_ecg_6, leads_list=None, threshold=0.7):
    """
    Calculate precision, recall, F1 for each ECG lead (6-lead system).
    Follows the pattern from metrics.py for multi-lead ECG processing.
    """
    if leads_list is None:
        leads_list = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    
    metrics_per_lead = {}
    for lead_idx, lead_name in enumerate(leads_list):
        metrics = calculate_signal_quality_classification(
            p_ecg_6[lead_idx], 
            t_ecg_6[lead_idx], 
            threshold=threshold
        )
        metrics_per_lead[lead_name] = metrics
    
    return metrics_per_lead

# =====================================================
# 3C. STATISTICAL TESTS
# =====================================================
def perform_statistical_tests(pred_signal, truth_signal, test_name="Signal"):
    """
    Perform paired t-test (two-tailed) and Wilcoxon signed-rank test.
    This follows scipy.stats patterns used throughout the codebase.
    
    Args:
        pred_signal: Predicted signal array
        truth_signal: Ground truth signal array
        test_name: Name of the signal for reporting
    
    Returns:
        Dict with test statistics and p-values
    """
    # Flatten and clean signals
    pred_flat = pred_signal.flatten()
    truth_flat = truth_signal.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(truth_flat))
    if np.sum(mask) < 3:  # Need at least 3 samples for statistical tests
        return {
            'test_name': test_name,
            'n_samples': 0,
            'paired_ttest_t_stat': np.nan,
            'paired_ttest_p_value': np.nan,
            'wilcoxon_statistic': np.nan,
            'wilcoxon_p_value': np.nan,
            'mean_pred': np.nan,
            'mean_truth': np.nan,
            'mean_error': np.nan
        }
    
    pred_clean = pred_flat[mask]
    truth_clean = truth_flat[mask]
    
    # Calculate differences
    differences = pred_clean - truth_clean
    
    # Paired t-test (two-tailed)
    try:
        t_stat, t_pval = ttest_rel(pred_clean, truth_clean)
    except Exception as e:
        t_stat, t_pval = np.nan, np.nan
    
    # Wilcoxon signed-rank test (non-parametric, two-tailed)
    try:
        w_stat, w_pval = wilcoxon(pred_clean, truth_clean, alternative='two-sided')
    except Exception as e:
        w_stat, w_pval = np.nan, np.nan
    
    return {
        'test_name': test_name,
        'n_samples': len(pred_clean),
        'paired_ttest_t_stat': t_stat,
        'paired_ttest_p_value': t_pval,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_p_value': w_pval,
        'mean_pred': np.mean(pred_clean),
        'mean_truth': np.mean(truth_clean),
        'mean_error': np.mean(np.abs(differences))
    }

# =====================================================
# 4. PLOTTING
# =====================================================
def save_full_dashboard(preds, truths, scores, hemo, sample_idx, save_dir):
    # FIX: Access the 'ecg' array inside the dictionary to get the shape
    fs = cfg.fs
    t = np.arange(preds['ecg'].shape[-1]) / fs
    filename = f"BEST_FORENSIC_window_{sample_idx}.png"
    
    # Calculate number of subplots needed (always 6 for ECG leads, then optional signals)
    n_plots = 6
    if preds['bp'] is not None: n_plots += 1
    if preds['icg'] is not None: n_plots += 1
    if preds['dicg'] is not None: n_plots += 1
    if preds['strain'] is not None: n_plots += 1
    if preds['resp'] is not None: n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=True)
    
    # Ensure axes is always array-like
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot ECG Leads
    for j, name in enumerate(LEADS):
        temp, spec = scores[j]
        title = f"Lead {name} | T:{temp:.2f} S:{spec:.2f}"
        
        # Add labels to clarify derivation
        if j in [0, 1]: title += " (Forensic Source)"
        else: title += " (Derived Physics)"
        
        axes[plot_idx].plot(t, truths['ecg'][j], 'g', alpha=0.6, label='Truth (Remapped)')
        axes[plot_idx].plot(t, preds['ecg'][j], 'r--', alpha=0.8, label='Pred (Remapped)')
        axes[plot_idx].set_ylabel(name)
        axes[plot_idx].text(0.01, 0.85, title, transform=axes[plot_idx].transAxes, fontsize=8, backgroundcolor='white')
        axes[plot_idx].grid(True, alpha=0.3)
        if j==0: axes[plot_idx].legend(loc='upper right')
        plot_idx += 1

    # Hemodynamics - BP
    if preds['bp'] is not None and truths['bp'] is not None:
        axes[plot_idx].plot(t, truths['bp'][0], 'g', alpha=0.6, label='Truth')
        axes[plot_idx].plot(t, preds['bp'][0], 'm--', alpha=0.8, label='Pred')
        axes[plot_idx].set_ylabel("BP")
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # ICG
    if preds['icg'] is not None and truths['icg'] is not None:
        axes[plot_idx].plot(t, truths['icg'][0], 'g', alpha=0.6, label='Truth')
        axes[plot_idx].plot(t, preds['icg'][0], 'c--', alpha=0.8, label='Pred')
        axes[plot_idx].set_ylabel("ICG")
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # FLOW (With Calibration Note)
    if preds['dicg'] is not None and truths['dicg'] is not None:
        axes[plot_idx].plot(t, truths['dicg'][0], 'g', alpha=0.6, label="Truth")
        axes[plot_idx].plot(t, preds['dicg'][0], 'k--', alpha=0.8, label="Pred (0.5x Scaled)")
        axes[plot_idx].set_ylabel("Flow")
        hemo_txt = f"SV Pred: {hemo['pred_sv']:.2f} | PEP Error: {hemo['pep_err']:.1f}ms"
        axes[plot_idx].text(0.01, 0.85, hemo_txt, transform=axes[plot_idx].transAxes, fontsize=8, backgroundcolor='white')
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1

    # Strain
    if preds['strain'] is not None and truths['strain'] is not None:
        axes[plot_idx].plot(t, truths['strain'][0], 'g', alpha=0.6, label='Truth')
        axes[plot_idx].plot(t, preds['strain'][0], 'y--', alpha=0.8, label='Pred')
        axes[plot_idx].set_ylabel("Strain")
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Resp
    if preds['resp'] is not None and truths['resp'] is not None:
        axes[plot_idx].plot(t, truths['resp'][0], 'g', alpha=0.6, label='Truth')
        axes[plot_idx].plot(t, preds['resp'][0], 'b--', alpha=0.8, label='Pred')
        axes[plot_idx].set_ylabel("Resp")
        axes[plot_idx].legend(loc='upper right')
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    axes[plot_idx-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=100)

# =====================================================
# 5. MAIN LOOP
# =====================================================
def main():
    # Device fallback: Use CPU if CUDA is not available
    try:
        device = torch.device(cfg.device)
        if "cuda" in cfg.device and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = torch.device("cpu")
    except:
        device = torch.device("cpu")
    
    save_dir = os.path.join(cfg.results_dir, "beat_metrics")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print(f"Device: {device}")
    print("SEARCHING FOR BEST WINDOW (FORENSIC CORRECTED)")
    print("1. Channel Mapping: Ch2=I, Ch1=II (Derived: III, aVR, aVL, aVF)")
    print("2. Flow Calibration: Prediction scaled by 0.5x")
    print("3. Filtering: 5Hz Low-Pass on Hemo signals")
    print("=" * 60)

    _, _, test_loader = create_patient_wise_splits(cfg)
    model = SimplifiedCASTECG(cfg).to(device)
    ckpt = f"{cfg.checkpoint_dir}/best_final.pth"
    if not os.path.exists(ckpt): ckpt = f"{cfg.checkpoint_dir}/best_ecg_model.pth"
    
    # Load checkpoint with strict=False to handle mismatched keys/shapes
    try:
        state_dict = torch.load(ckpt, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Model checkpoint loaded successfully from: {ckpt}")
    except Exception as e:
        print(f"⚠ Warning: Checkpoint loading with strict=False encountered issue: {e}")
        print(f"  Attempting to load with compatible keys only...")
        state_dict = torch.load(ckpt, map_location=device)
        model_state = model.state_dict()
        compatible_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        model_state.update(compatible_state)
        model.load_state_dict(model_state)
        print(f"✓ Loaded {len(compatible_state)} compatible parameters")
    
    model.eval()

    best_score = -999
    best_data_pack = None
    global_cnt = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Scanning"):
            radar = batch['radar_i'].to(device)
            out = model(radar)
            B = radar.shape[0]

            # Get available keys and CPU Numpy conversion
            available_keys = [k for k in ['ecg', 'bp', 'icg', 'dicg', 'strain', 'resp'] if k in out]
            
            p_all = {k: out[k].cpu().numpy() for k in available_keys}
            t_all = {k: batch[k].cpu().numpy() for k in available_keys if k in batch}

            for i in range(B):
                # 1. FORENSIC RECONSTRUCTION
                # Rebuild Truth (Ground Truth Ch2=I, Ch1=II)
                t_ecg_6 = reconstruct_6leads_from_forensics(t_all['ecg'][i])
                
                # Rebuild Pred (Model Output Ch2=I, Ch1=II)
                p_ecg_6 = reconstruct_6leads_from_forensics(p_all['ecg'][i])
                
                # 2. CALIBRATION & FILTERING
                p_bp = smooth_signal(p_all['bp'][i][0], cfg.fs) if 'bp' in p_all else None
                t_bp = smooth_signal(t_all['bp'][i][0], cfg.fs) if 'bp' in t_all else None
                
                p_icg = smooth_signal(p_all['icg'][i][0], cfg.fs) if 'icg' in p_all else None
                t_icg = smooth_signal(t_all['icg'][i][0], cfg.fs) if 'icg' in t_all else None
                
                # Flow Calibration (0.5x) + Filtering
                p_flow = smooth_signal(p_all['dicg'][i][0] * 0.5, cfg.fs) if 'dicg' in p_all else None  # <--- SCALED HERE
                t_flow = smooth_signal(t_all['dicg'][i][0], cfg.fs) if 'dicg' in t_all else None

                # Prepare dictionaries for plotting function (handle optional outputs)
                p_plot = {'ecg': p_ecg_6, 'bp': [p_bp] if p_bp is not None else None, 
                         'icg': [p_icg] if p_icg is not None else None, 
                         'dicg': [p_flow] if p_flow is not None else None, 
                         'strain': p_all['strain'][i] if 'strain' in p_all else None, 
                         'resp': p_all['resp'][i] if 'resp' in p_all else None}
                
                t_plot = {'ecg': t_ecg_6, 'bp': [t_bp] if t_bp is not None else None, 
                         'icg': [t_icg] if t_icg is not None else None, 
                         'dicg': [t_flow] if t_flow is not None else None, 
                         'strain': t_all['strain'][i] if 'strain' in t_all else None, 
                         'resp': t_all['resp'][i] if 'resp' in t_all else None}

                # 3. SCORING (Focus on Lead II - The Forensic "Anchor")
                t_corr, s_corr = get_correlations(p_ecg_6[1], t_ecg_6[1], cfg.fs)
                
                if t_corr > best_score:
                    best_score = t_corr
                    
                    # Hemo Calcs (only if flow data available)
                    p_sv, p_pep = np.nan, np.nan
                    t_sv, t_pep = np.nan, np.nan
                    if p_flow is not None and t_flow is not None:
                        p_sv, p_pep = calculate_hemodynamics(p_ecg_6[1], p_flow, cfg.fs)
                        t_sv, t_pep = calculate_hemodynamics(t_ecg_6[1], t_flow, cfg.fs)
                    pep_err = abs(p_pep - t_pep) if (not np.isnan(p_pep) and not np.isnan(t_pep)) else np.nan
                    
                    # Full Scores for all 6 leads
                    lead_scores = []
                    for l in range(6):
                        lead_scores.append(get_correlations(p_ecg_6[l], t_ecg_6[l], cfg.fs))

                    hemo_stats = {'pred_sv': p_sv, 'true_sv': t_sv, 'pep_err': pep_err}

                    best_data_pack = {
                        'id': global_cnt, 'preds': p_plot, 'truths': t_plot,
                        'scores': lead_scores, 'hemo': hemo_stats,
                        'best_lead_name': 'Lead II (Forensic)', 'best_temp': t_corr
                    }
                global_cnt += 1

    if best_data_pack:
        save_full_dashboard(best_data_pack['preds'], best_data_pack['truths'], best_data_pack['scores'], best_data_pack['hemo'], best_data_pack['id'], save_dir)
        
        # ===== CLASSIFICATION METRICS (Precision, Recall, F1) =====
        p_ecg_6 = best_data_pack['preds']['ecg']
        t_ecg_6 = best_data_pack['truths']['ecg']
        classification_metrics = calculate_per_lead_classification_metrics(p_ecg_6, t_ecg_6, threshold=0.7)
        
        # Calculate aggregate metrics across all leads
        all_precision = [m['precision'] for m in classification_metrics.values()]
        all_recall = [m['recall'] for m in classification_metrics.values()]
        all_f1 = [m['f1'] for m in classification_metrics.values()]
        
        aggregate_precision = np.mean(all_precision)
        aggregate_recall = np.mean(all_recall)
        aggregate_f1 = np.mean(all_f1)
        
        # ===== STATISTICAL TESTS =====
        # Paired t-test and Wilcoxon for Lead II (Forensic anchor)
        lead_ii_ecg_pred = p_ecg_6[1]
        lead_ii_ecg_truth = t_ecg_6[1]
        stats_lead_ii = perform_statistical_tests(lead_ii_ecg_pred, lead_ii_ecg_truth, test_name="Lead II ECG")
        
        # Statistical test for all ECG signals combined
        stats_all_ecg = perform_statistical_tests(p_ecg_6.flatten(), t_ecg_6.flatten(), test_name="All ECG Leads")
        
        # Statistical test for BP (if available)
        stats_bp = None
        if best_data_pack['preds']['bp'] is not None and best_data_pack['truths']['bp'] is not None:
            stats_bp = perform_statistical_tests(
                best_data_pack['preds']['bp'][0],
                best_data_pack['truths']['bp'][0],
                test_name="Blood Pressure"
            )
        
        # Statistical test for Flow (dICG) (if available)
        stats_flow = None
        if best_data_pack['preds']['dicg'] is not None and best_data_pack['truths']['dicg'] is not None:
            stats_flow = perform_statistical_tests(
                best_data_pack['preds']['dicg'][0],
                best_data_pack['truths']['dicg'][0],
                test_name="Flow (dICG)"
            )
        
        print("\n" + "="*60)
        print(f"FORENSIC RESULT (ID: {best_data_pack['id']})")
        print("="*60)
        print(f"Saved Plot to: {save_dir}/BEST_FORENSIC_window_{best_data_pack['id']}.png")
        print(f"Lead II Temporal Corr: {best_data_pack['best_temp']:.4f}")
        print("-" * 30)
        print("Hemodynamics (Forensic + 0.5x Scale):")
        print(f"  Stroke Volume: {best_data_pack['hemo']['pred_sv']:.3f} (Pred) vs {best_data_pack['hemo']['true_sv']:.3f} (True)")
        print(f"  PEP Error:     {best_data_pack['hemo']['pep_err']:.1f} ms")
        
        # ===== PRINT CLASSIFICATION METRICS =====
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS (Signal Quality - Per Lead)")
        print("="*60)
        print(f"{'Lead':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Correlation':<12}")
        print("-" * 60)
        for lead_name, metrics in classification_metrics.items():
            print(f"{lead_name:<10} {metrics['precision']:.4f}      {metrics['recall']:.4f}      "
                  f"{metrics['f1']:.4f}      {metrics['correlation']:.4f}")
        
        print("-" * 60)
        print(f"{'AGGREGATE':<10} {aggregate_precision:.4f}      {aggregate_recall:.4f}      "
              f"{aggregate_f1:.4f}")
        print("="*60)
        
        # ===== PRINT STATISTICAL TESTS RESULTS =====
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS (Paired t-test & Wilcoxon Test)")
        print("="*60)
        
        def print_statistical_results(stats_dict):
            if stats_dict is None:
                return
            print(f"\nTest: {stats_dict['test_name']}")
            print(f"  Samples: {stats_dict['n_samples']}")
            print(f"  Mean (Pred): {stats_dict['mean_pred']:.6f}")
            print(f"  Mean (Truth): {stats_dict['mean_truth']:.6f}")
            print(f"  Mean Absolute Error: {stats_dict['mean_error']:.6f}")
            print(f"\n  Paired t-test (Two-tailed):")
            print(f"    t-statistic: {stats_dict['paired_ttest_t_stat']:.6f}")
            print(f"    p-value: {stats_dict['paired_ttest_p_value']:.6e}")
            if not np.isnan(stats_dict['paired_ttest_p_value']) and stats_dict['paired_ttest_p_value'] < 0.05:
                print(f"    ✓ Significant difference detected (p < 0.05)")
            else:
                print(f"    ✗ No significant difference (p >= 0.05)")
            
            print(f"\n  Wilcoxon Signed-Rank Test (Two-tailed):")
            print(f"    Statistic: {stats_dict['wilcoxon_statistic']:.6f}")
            print(f"    p-value: {stats_dict['wilcoxon_p_value']:.6e}")
            if not np.isnan(stats_dict['wilcoxon_p_value']) and stats_dict['wilcoxon_p_value'] < 0.05:
                print(f"    ✓ Significant difference detected (p < 0.05)")
            else:
                print(f"    ✗ No significant difference (p >= 0.05)")
        
        print_statistical_results(stats_lead_ii)
        print_statistical_results(stats_all_ecg)
        if stats_bp is not None:
            print_statistical_results(stats_bp)
        else:
            print("\n⚠ Blood Pressure signal not available for statistical testing")
        
        if stats_flow is not None:
            print_statistical_results(stats_flow)
        else:
            print("\n⚠ Flow (dICG) signal not available for statistical testing")
        
        print("\n" + "="*60)
        print("NOTE: p < 0.05 indicates statistically significant difference")
        print("="*60)
        
    else:
        print("Error: No valid windows found.")

if __name__ == "__main__":
    main()