import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import welch, find_peaks, butter, filtfilt

# Project Imports
from configs.config import cfg
from models.cast_ecg import SimplifiedCASTECG_Paper as SimplifiedCASTECG
from dataload.dataset import create_patient_wise_splits
from utils.xai import GradCAM, visualize_attention

# --- CONFIGURATION ---
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']

# =====================================================
# 1. PHYSICS & MAPPING ENGINE
# =====================================================
def reconstruct_6leads_from_forensics(raw_data_3ch):
    """
    Takes the raw 3-channel output/truth and maps it correctly based on forensic analysis:
    - Channel 2 -> Lead I
    - Channel 1 -> Lead II
    - Channel 0 -> Lead III (Ignored, we recalculate it)
    """
    # 1. Extract the reliable leads based on forensic finding
    # Shape of raw_data_3ch is usually [Leads, Time]
    
    lead_I = raw_data_3ch[2]  # Forensic Result: Index 2 is Lead I
    lead_II = raw_data_3ch[1] # Forensic Result: Index 1 is Lead II
    
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
# 4. XAI ENHANCED PLOTTING
# =====================================================
def save_xai_enhanced_dashboard(preds, truths, scores, hemo, sample_idx, save_dir, radar_sample, cam_map):
    """
    Enhanced dashboard with XAI visualization integrated
    """
    fs = cfg.fs
    t = np.arange(preds['ecg'].shape[-1]) / fs
    filename = f"MASTER_XAI_window_{sample_idx}.png"
    
    # Create a larger figure to accommodate XAI
    fig, axes = plt.subplots(13, 1, figsize=(14, 28), sharex=True)
    
    # XAI Panel 1: Radar Magnitude with Attention Overlay
    radar_mag = radar_sample[1].cpu().numpy()
    cam_resized = np.interp(np.linspace(0, 1, len(radar_mag)), 
                           np.linspace(0, 1, len(cam_map)), 
                           cam_map)
    
    axes[0].plot(t, radar_mag, 'k', alpha=0.3, label='Radar Mag')
    axes[0].scatter(t, radar_mag, c=cam_resized, cmap='jet', s=3, alpha=0.8)
    axes[0].set_ylabel("Radar + XAI")
    axes[0].set_title("XAI: Model Attention on Radar Input (Red = High Attention)")
    axes[0].legend(loc='upper right')
    
    # XAI Panel 2: Attention Heatmap
    axes[1].plot(t, cam_resized, 'r-', linewidth=2)
    axes[1].fill_between(t, 0, cam_resized, alpha=0.3, color='red')
    axes[1].set_ylabel("Attention")
    axes[1].set_title("Grad-CAM Attention Map")
    axes[1].grid(True, alpha=0.3)
    
    # Plot ECG Leads with attention indicators
    for j, name in enumerate(LEADS):
        temp, spec = scores[j]
        title = f"Lead {name} | T:{temp:.2f} S:{spec:.2f}"
        
        # Add labels to clarify derivation
        if j in [0, 1]: title += " (Forensic Source)"
        else: title += " (Derived Physics)"
        
        # Add attention peaks as vertical lines
        from scipy.signal import find_peaks
        cam_peaks, _ = find_peaks(cam_resized, height=0.5, distance=50)
        
        axes[j+2].plot(t, truths['ecg'][j], 'g', alpha=0.6, label='Truth (Remapped)')
        axes[j+2].plot(t, preds['ecg'][j], 'r--', alpha=0.8, label='Pred (Remapped)')
        axes[j+2].set_ylabel(name)
        axes[j+2].text(0.01, 0.85, title, transform=axes[j+2].transAxes, fontsize=8, backgroundcolor='white')
        axes[j+2].grid(True, alpha=0.3)
        
        # Mark attention peaks
        for p in cam_peaks:
            axes[j+2].axvline(x=t[p], color='orange', linestyle=':', alpha=0.5, linewidth=1)
        
        if j==0: axes[j+2].legend(loc='upper right')

    # Hemodynamics
    axes[8].plot(t, truths['bp'][0], 'g', alpha=0.6); axes[8].plot(t, preds['bp'][0], 'm--', alpha=0.8); axes[8].set_ylabel("BP")
    axes[9].plot(t, truths['icg'][0], 'g', alpha=0.6); axes[9].plot(t, preds['icg'][0], 'c--', alpha=0.8); axes[9].set_ylabel("ICG")
    
    # FLOW (With Calibration Note)
    axes[10].plot(t, truths['dicg'][0], 'g', alpha=0.6, label="Truth")
    axes[10].plot(t, preds['dicg'][0], 'k--', alpha=0.8, label="Pred (0.5x Scaled)")
    axes[10].set_ylabel("Flow")
    hemo_txt = f"SV Pred: {hemo['pred_sv']:.2f} | PEP Error: {hemo['pep_err']:.1f}ms"
    axes[10].text(0.01, 0.85, hemo_txt, transform=axes[10].transAxes, fontsize=8, backgroundcolor='white')
    axes[10].legend(loc='upper right')

    axes[11].plot(t, truths['strain'][0], 'g', alpha=0.6); axes[11].plot(t, preds['strain'][0], 'y--', alpha=0.8); axes[11].set_ylabel("Strain")
    axes[12].plot(t, truths['resp'][0], 'g', alpha=0.6); axes[12].plot(t, preds['resp'][0], 'b--', alpha=0.8); axes[12].set_ylabel("Resp")
    axes[12].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=150)
    plt.close()

def save_xai_detailed_analysis(radar_sample, ecg_pred, ecg_truth, cam_map, sample_idx, save_dir):
    """
    Detailed XAI analysis with multiple views
    """
    fs = cfg.fs
    t = np.arange(len(ecg_truth)) / fs
    
    # Create detailed XAI figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Panel 1: Raw Radar with attention overlay
    radar_mag = radar_sample[1].cpu().numpy()
    cam_resized = np.interp(np.linspace(0, 1, len(radar_mag)), 
                           np.linspace(0, 1, len(cam_map)), 
                           cam_map)
    
    axes[0].plot(t, radar_mag, 'k', alpha=0.3, label='Radar Mag')
    scatter = axes[0].scatter(t, radar_mag, c=cam_resized, cmap='jet', s=5)
    axes[0].set_ylabel("Radar Mag")
    axes[0].set_title("XAI: Radar Signal with Model Attention Overlay")
    plt.colorbar(scatter, ax=axes[0], label='Attention Weight')
    
    # Panel 2: Attention heatmap
    axes[1].plot(t, cam_resized, 'r-', linewidth=2)
    axes[1].fill_between(t, 0, cam_resized, alpha=0.3, color='red')
    axes[1].set_ylabel("Attention")
    axes[1].set_title("Grad-CAM Attention Map")
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: ECG Prediction vs Truth
    axes[2].plot(t, ecg_truth, 'g', alpha=0.7, label='Truth', linewidth=2)
    axes[2].plot(t, ecg_pred, 'r--', alpha=0.8, label='Prediction', linewidth=2)
    axes[2].set_ylabel("ECG Lead II")
    axes[2].set_title("ECG Lead II: Truth vs Prediction")
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    # Panel 4: Combined view with attention markers
    axes[3].plot(t, radar_mag, 'k', alpha=0.2, label='Radar')
    axes[3].plot(t, ecg_truth, 'g', alpha=0.7, label='ECG Truth')
    axes[3].plot(t, ecg_pred, 'r--', alpha=0.8, label='ECG Pred')
    
    # Mark high attention regions
    high_attention_idx = np.where(cam_resized > 0.6)[0]
    if len(high_attention_idx) > 0:
        axes[3].scatter(t[high_attention_idx], np.zeros_like(high_attention_idx), 
                       c='red', s=20, alpha=0.6, label='High Attention')
    
    axes[3].set_ylabel("Combined")
    axes[3].set_title("Combined View with High Attention Markers")
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"XAI_DETAILED_{sample_idx}.png"), dpi=150)
    plt.close()

def detect_ecg_waves(ecg_signal, fs):
    """
    Detect P, QRS, and T waves in ECG signal
    Returns dictionary with wave positions and types
    """
    from scipy.signal import find_peaks
    
    # Find R peaks (most prominent)
    r_peaks, _ = find_peaks(ecg_signal, distance=fs*0.5, prominence=0.5)
    
    waves = []
    
    for r_peak in r_peaks:
        # P wave: before R peak (200-300ms window)
        p_window_start = max(0, r_peak - int(0.3 * fs))
        p_window_end = r_peak - int(0.05 * fs)
        if p_window_end > p_window_start:
            p_segment = ecg_signal[p_window_start:p_window_end]
            if len(p_segment) > 0:
                p_peak = p_window_start + np.argmax(p_segment)
                waves.append((p_peak, 'P', 'blue'))
        
        # Q wave: just before R peak
        q_window_start = max(0, r_peak - int(0.08 * fs))
        q_window_end = r_peak
        if q_window_end > q_window_start:
            q_segment = ecg_signal[q_window_start:q_window_end]
            if len(q_segment) > 0:
                q_peak = q_window_start + np.argmin(q_segment)
                waves.append((q_peak, 'Q', 'orange'))
        
        # R peak
        waves.append((r_peak, 'R', 'red'))
        
        # S wave: just after R peak
        s_window_start = r_peak
        s_window_end = min(len(ecg_signal), r_peak + int(0.08 * fs))
        if s_window_end > s_window_start:
            s_segment = ecg_signal[s_window_start:s_window_end]
            if len(s_segment) > 0:
                s_peak = s_window_start + np.argmin(s_segment)
                waves.append((s_peak, 'S', 'yellow'))
        
        # T wave: after R peak (200-400ms window)
        t_window_start = r_peak + int(0.1 * fs)
        t_window_end = min(len(ecg_signal), r_peak + int(0.5 * fs))
        if t_window_end > t_window_start:
            t_segment = ecg_signal[t_window_start:t_window_end]
            if len(t_segment) > 0:
                t_peak = t_window_start + np.argmax(t_segment)
                waves.append((t_peak, 'T', 'green'))
    
    return sorted(waves, key=lambda x: x[0])

def color_ecg_waveform(ecg_signal, waves, fs):
    """
    Color ECG waveform segments based on P-QRS-T waves
    Returns colored segments for plotting
    """
    if not waves:
        return [(0, len(ecg_signal), 'black')]  # Default if no waves detected
    
    segments = []
    prev_pos = 0
    
    for i, (wave_pos, wave_type, color) in enumerate(waves):
        # Add segment before this wave
        if wave_pos > prev_pos:
            segments.append((prev_pos, wave_pos, 'black'))
        
        # Determine wave segment boundaries
        if wave_type == 'P':
            # P wave segment
            end_pos = waves[i+1][0] if i+1 < len(waves) else min(len(ecg_signal), wave_pos + int(0.1 * fs))
            segments.append((wave_pos, end_pos, color))
        elif wave_type == 'Q':
            # Q wave segment (short)
            end_pos = waves[i+1][0] if i+1 < len(waves) else min(len(ecg_signal), wave_pos + int(0.05 * fs))
            segments.append((wave_pos, end_pos, color))
        elif wave_type == 'R':
            # R wave segment (sharp peak)
            end_pos = waves[i+1][0] if i+1 < len(waves) else min(len(ecg_signal), wave_pos + int(0.08 * fs))
            segments.append((wave_pos, end_pos, color))
        elif wave_type == 'S':
            # S wave segment (short)
            end_pos = waves[i+1][0] if i+1 < len(waves) else min(len(ecg_signal), wave_pos + int(0.08 * fs))
            segments.append((wave_pos, end_pos, color))
        elif wave_type == 'T':
            # T wave segment
            end_pos = waves[i+1][0] if i+1 < len(waves) else min(len(ecg_signal), wave_pos + int(0.3 * fs))
            segments.append((wave_pos, end_pos, color))
        
        prev_pos = end_pos
    
    # Add remaining segment
    if prev_pos < len(ecg_signal):
        segments.append((prev_pos, len(ecg_signal), 'black'))
    
    return segments

def save_comprehensive_waveform_analysis(preds, truths, sample_idx, save_dir):
    """
    Comprehensive waveform analysis with colored ECG waves and hemodynamic signals
    """
    fs = cfg.fs
    t = np.arange(preds['ecg'].shape[-1]) / fs
    
    # Create figure with subplots for all leads + hemodynamics
    fig, axes = plt.subplots(9, 1, figsize=(16, 20), sharex=True)
    fig.suptitle(f'Comprehensive Waveform Analysis - Sample {sample_idx}', fontsize=16, fontweight='bold')
    
    # Plot all 6 ECG leads with colored P-QRS-T waves
    for i, lead_name in enumerate(LEADS):
        ax = axes[i]
        
        # Detect waves for PREDICTED signal (not truth)
        waves = detect_ecg_waves(preds['ecg'][i], fs)
        
        # Get colored segments for PREDICTED signal
        pred_segments = color_ecg_waveform(preds['ecg'][i], waves, fs)
        
        # Plot PREDICTED signal with colored segments
        for start_pos, end_pos, color in pred_segments:
            segment_t = t[start_pos:end_pos]
            segment_signal = preds['ecg'][i][start_pos:end_pos]
            ax.plot(segment_t, segment_signal, color=color, linewidth=2, alpha=0.8)
        
        # Add wave labels at key positions for PREDICTED signal
        for wave_pos, wave_type, color in waves:
            if wave_type in ['R', 'P', 'T']:  # Only label major waves
                ax.text(t[wave_pos], preds['ecg'][i][wave_pos] + 0.15, wave_type,
                       fontsize=9, fontweight='bold', ha='center', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Styling
        ax.set_ylabel(f'{lead_name}\n(mV)', fontsize=10)
        ax.set_title(f'Lead {lead_name} - P-QRS-T Wave Analysis (Predicted)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add wave legend
        if i == 0:
            legend_text = ('Wave Colors:\n'
                        '━━ P Wave (Blue)\n'
                        '━━ Q Wave (Orange)\n'
                        '━━ R Wave (Red)\n'
                        '━━ S Wave (Yellow)\n'
                        '━━ T Wave (Green)\n'
                        '━━ Baseline (Black)')
            ax.text(0.02, 0.95, legend_text,
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Plot Blood Pressure with coloring
    ax_bp = axes[6]
    ax_bp.plot(t, truths['bp'][0], 'purple', alpha=0.8, linewidth=2.5, label='Truth BP')
    ax_bp.plot(t, preds['bp'][0], 'r--', alpha=0.6, linewidth=1.5, label='Prediction BP')
    ax_bp.fill_between(t, truths['bp'][0], preds['bp'][0], alpha=0.2, color='purple')
    ax_bp.set_ylabel('BP\n(mmHg)', fontsize=10)
    ax_bp.set_title('Blood Pressure Analysis', fontsize=11, fontweight='bold', color='purple')
    ax_bp.grid(True, alpha=0.3)
    ax_bp.legend(loc='upper right', fontsize=8)
    
    # Plot ICG with coloring
    ax_icg = axes[7]
    ax_icg.plot(t, truths['icg'][0], 'darkblue', alpha=0.8, linewidth=2.5, label='Truth ICG')
    ax_icg.plot(t, preds['icg'][0], 'r--', alpha=0.6, linewidth=1.5, label='Prediction ICG')
    ax_icg.fill_between(t, truths['icg'][0], preds['icg'][0], alpha=0.2, color='cyan')
    ax_icg.set_ylabel('ICG\n(Ω)', fontsize=10)
    ax_icg.set_title('Impedance Cardiography (ICG) Analysis', fontsize=11, fontweight='bold', color='darkblue')
    ax_icg.grid(True, alpha=0.3)
    ax_icg.legend(loc='upper right', fontsize=8)
    
    # Plot Flow with coloring
    ax_flow = axes[8]
    ax_flow.plot(t, truths['dicg'][0], 'darkorange', alpha=0.8, linewidth=2.5, label='Truth Flow')
    ax_flow.plot(t, preds['dicg'][0], 'r--', alpha=0.6, linewidth=1.5, label='Prediction Flow')
    ax_flow.fill_between(t, truths['dicg'][0], preds['dicg'][0], alpha=0.2, color='orange')
    ax_flow.set_ylabel('Flow\n(mL/s)', fontsize=10)
    ax_flow.set_title('Blood Flow Analysis', fontsize=11, fontweight='bold', color='darkorange')
    ax_flow.grid(True, alpha=0.3)
    ax_flow.legend(loc='upper right', fontsize=8)
    ax_flow.set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"COMPREHENSIVE_WAVES_{sample_idx}.png"), dpi=150, bbox_inches='tight')
    plt.close()

# =====================================================
# 5. MAIN LOOP WITH XAI INTEGRATION
# =====================================================
def main():
    # Handle device selection properly
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU (this will be slower)")
    
    save_dir = os.path.join(cfg.results_dir, "master_xai_results")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("MASTER TEST WITH XAI INTEGRATION")
    print("=" * 80)
    print("1. Channel Mapping: Ch2=I, Ch1=II (Derived: III, aVR, aVL, aVF)")
    print("2. Flow Calibration: Prediction scaled by 0.5x")
    print("3. Filtering: 5Hz Low-Pass on Hemo signals")
    print("4. XAI: Grad-CAM attention visualization integrated")
    print("5. Enhanced visualizations with attention overlays")
    print("=" * 80)

    _, _, test_loader = create_patient_wise_splits(cfg)
    model = SimplifiedCASTECG(cfg).to(device)
    ckpt = f"{cfg.checkpoint_dir}/best_final.pth"
    if not os.path.exists(ckpt): ckpt = f"{cfg.checkpoint_dir}/best_ecg_model.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Initialize XAI (Grad-CAM) - target the input convolution for better feature representation
    target_layer = model.in_conv  # Use input convolution as target layer
    grad_cam = GradCAM(model, target_layer)

    best_score = -999
    best_data_pack = None
    global_cnt = 0

    print("\nScanning test set with XAI analysis...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Scanning with XAI"):
            radar = batch['radar_i'].to(device)
            out = model(radar)
            B = radar.shape[0]

            # CPU Numpy
            p_all = {k: out[k].cpu().numpy() for k in ['ecg', 'bp', 'icg', 'dicg', 'strain', 'resp']}
            t_all = {k: batch[k].cpu().numpy() for k in ['ecg', 'bp', 'icg', 'dicg', 'strain', 'resp']}

            for i in range(B):
                # 1. FORENSIC RECONSTRUCTION
                t_ecg_6 = reconstruct_6leads_from_forensics(t_all['ecg'][i])
                p_ecg_6 = reconstruct_6leads_from_forensics(p_all['ecg'][i])
                
                # 2. CALIBRATION & FILTERING
                p_bp = smooth_signal(p_all['bp'][i][0], cfg.fs)
                t_bp = smooth_signal(t_all['bp'][i][0], cfg.fs)
                
                p_icg = smooth_signal(p_all['icg'][i][0], cfg.fs)
                t_icg = smooth_signal(t_all['icg'][i][0], cfg.fs)
                
                # Flow Calibration (0.5x) + Filtering
                p_flow = smooth_signal(p_all['dicg'][i][0] * 0.5, cfg.fs)
                t_flow = smooth_signal(t_all['dicg'][i][0], cfg.fs)

                # Prepare dictionaries for plotting function
                p_plot = {'ecg': p_ecg_6, 'bp': [p_bp], 'icg': [p_icg], 'dicg': [p_flow], 'strain': p_all['strain'][i], 'resp': p_all['resp'][i]}
                t_plot = {'ecg': t_ecg_6, 'bp': [t_bp], 'icg': [t_icg], 'dicg': [t_flow], 'strain': t_all['strain'][i], 'resp': t_all['resp'][i]}

                # 3. XAI ANALYSIS
                try:
                    # Enable gradients for XAI analysis
                    with torch.enable_grad():
                        radar_sample = radar[i:i+1].clone().detach().requires_grad_(True)
                        # Get Grad-CAM attention map
                        cam_map, ecg_pred = grad_cam(radar_sample)
                    
                    # 4. SCORING (Focus on Lead II - The Forensic "Anchor")
                    t_corr, s_corr = get_correlations(p_ecg_6[1], t_ecg_6[1], cfg.fs)
                    
                    if t_corr > best_score:
                        best_score = t_corr
                        
                        # Hemo Calcs
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
                            'best_lead_name': 'Lead II (Forensic)', 'best_temp': t_corr,
                            'radar_sample': radar[i], 'cam_map': cam_map,
                            'ecg_pred': ecg_pred, 'ecg_truth': t_ecg_6[1]
                        }
                except Exception as e:
                    print(f"XAI analysis failed for sample {global_cnt}: {e}")
                    # Still score without XAI
                    t_corr, s_corr = get_correlations(p_ecg_6[1], t_ecg_6[1], cfg.fs)
                    if t_corr > best_score:
                        best_score = t_corr
                        
                        # Hemo Calcs
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
                            'best_lead_name': 'Lead II (Forensic)', 'best_temp': t_corr,
                            'radar_sample': None, 'cam_map': None,
                            'ecg_pred': None, 'ecg_truth': t_ecg_6[1]
                        }
                
                global_cnt += 1

    # 5. SAVE RESULTS WITH XAI
    if best_data_pack:
        print(f"\n{'='*80}")
        print(f"MASTER XAI RESULT (ID: {best_data_pack['id']})")
        print(f"{'='*80}")
        
        # Save enhanced dashboard
        if best_data_pack['radar_sample'] is not None and best_data_pack['cam_map'] is not None:
            save_xai_enhanced_dashboard(
                best_data_pack['preds'], best_data_pack['truths'], 
                best_data_pack['scores'], best_data_pack['hemo'], 
                best_data_pack['id'], save_dir,
                best_data_pack['radar_sample'], best_data_pack['cam_map']
            )
            
            # Save detailed XAI analysis
            save_xai_detailed_analysis(
                best_data_pack['radar_sample'], best_data_pack['ecg_pred'], 
                best_data_pack['ecg_truth'], best_data_pack['cam_map'],
                best_data_pack['id'], save_dir
            )
            
            # Save comprehensive waveform analysis
            save_comprehensive_waveform_analysis(
                best_data_pack['preds'], best_data_pack['truths'],
                best_data_pack['id'], save_dir
            )
            
            print(f"✓ Enhanced XAI Dashboard: {save_dir}/MASTER_XAI_window_{best_data_pack['id']}.png")
            print(f"✓ Detailed XAI Analysis: {save_dir}/XAI_DETAILED_{best_data_pack['id']}.png")
            print(f"✓ Comprehensive Waveform Analysis: {save_dir}/COMPREHENSIVE_WAVES_{best_data_pack['id']}.png")
        else:
            # Fallback to original dashboard if XAI failed
            from final_test import save_full_dashboard
            save_full_dashboard(best_data_pack['preds'], best_data_pack['truths'], 
                               best_data_pack['scores'], best_data_pack['hemo'], 
                               best_data_pack['id'], save_dir)
            print(f"✓ Standard Dashboard: {save_dir}/BEST_FORENSIC_window_{best_data_pack['id']}.png")
        
        print(f"Lead II Temporal Corr: {best_data_pack['best_temp']:.4f}")
        print("-" * 40)
        print("Hemodynamics (Forensic + 0.5x Scale):")
        print(f"  Stroke Volume: {best_data_pack['hemo']['pred_sv']:.3f} (Pred) vs {best_data_pack['hemo']['true_sv']:.3f} (True)")
        print(f"  PEP Error:     {best_data_pack['hemo']['pep_err']:.1f} ms")
        print("-" * 40)
        print("XAI Features:")
        print("  ✓ Grad-CAM attention visualization")
        print("  ✓ Radar signal with attention overlay")
        print("  ✓ Attention peaks marked on ECG leads")
        print("  ✓ Detailed XAI analysis plots")
        print("  ✓ Comprehensive P-QRS-T wave analysis for all 6 leads")
        print("  ✓ Colored hemodynamic signals (BP, ICG, Flow)")
        print("  ✓ Single comprehensive waveform image")
        print(f"{'='*80}")
    else:
        print("Error: No valid windows found.")

if __name__ == "__main__":
    main()
