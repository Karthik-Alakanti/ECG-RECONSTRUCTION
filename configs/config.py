"""
Configuration: 7-Task Model with Multi-Band Input
"""
import torch

class Config:
    # --- Data parameters ---
    fs = 128.0 
    window_size_samples = 1310
    max_input_length = 1310
    stride_samples = 655

    # Paths
    data_root = '../../data'
    # New filename for 4-channel input
    h5_file_pattern = 'multiband_4ch_128hz_{m}.h5' 
    # maneuvers_to_load = [('1', 'Resting'), ('2', 'Valsalva'), ('3', 'Apnea')]
    maneuvers_to_load = [('1', 'Resting')]
    # Saving
    checkpoint_dir = 'checkpoints_multiband'
    results_dir = 'results_multiband_resting'
    figures_dir = 'results_multiband_resting'

    # --- Model Architecture ---
    in_channels = 4  # <--- CHANGED: Phase/Mag (Heart) + Phase/Mag (Resp)
    base_channels = 32
    n_filters = 8
    n_scales = 4
    
    # --- Training ---
    batch_size = 16         
    epochs = 100            
    lr = 1e-3               
    weight_decay = 1e-3
    grad_clip = 1.0
    
    # --- LOSS WEIGHTS ---
    lambda_l1 = 1.0      
    lambda_peak = 3.0       
    lambda_temp_corr = 2.5  
    lambda_spec_corr = 2.0  
    lambda_slope = 5.0      
    lambda_neg = 1.0       

    lambda_bp_l1 = 1.0
    lambda_bp_corr = 2.0
    
    lambda_icg_l1 = 1.0
    lambda_icg_corr = 2.0
    
    lambda_dicg_corr = 3.0 
    lambda_strain_l1 = 1.0 
    
    lambda_resp_l1 = 2.0   
    lambda_tv = 0.1        
    
    # Use CUDA if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 0

cfg = Config()