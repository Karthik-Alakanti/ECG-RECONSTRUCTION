"""
Explainable AI (XAI): 1D Grad-CAM for Radar-to-ECG
Visualizes which part of the Radar signal contributes most to the R-Peak prediction.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        
        # Hook for gradients
        target_layer.register_full_backward_hook(self.save_gradient)
        # Hook for activations
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def __call__(self, x):
        # Ensure input requires gradients
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
            
        # 1. Forward Pass
        self.model.zero_grad()
        output = self.model(x)
        
        # 2. Target: Maximizing the R-Peak (Lead II)
        # We want to explain what causes the highest peaks in Lead II
        ecg_out = output['ecg'][:, 1, :] # Lead II
        
        # Find location of max activation (R-peak)
        score = ecg_out.max()
        
        # 3. Backward Pass (Compute Gradients)
        score.backward(retain_graph=True)
        
        # 4. Generate CAM
        # Global Average Pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2])
        
        # Weight the activations by the gradients
        # activations: [1, Channels, Length]
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :] *= pooled_gradients[i]
            
        # Average the channels to get the heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU (We only care about positive influence)
        heatmap = F.relu(heatmap)
        
        # Normalize
        if torch.max(heatmap) > 1e-8:
            heatmap /= torch.max(heatmap)
        
        return heatmap.data.cpu().numpy(), ecg_out.data.cpu().numpy().flatten()

def visualize_attention(radar_sample, ecg_truth, cam_map, save_name):
    """
    Plots Radar (Mag), ECG (Truth), and the Attention Heatmap
    """
    t = np.arange(len(cam_map))
    
    # Resize CAM to match signal length if needed
    if len(cam_map) != len(ecg_truth):
        cam_map = np.interp(np.linspace(0, 1, len(ecg_truth)), 
                            np.linspace(0, 1, len(cam_map)), 
                            cam_map)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # 1. Radar Magnitude (Input)
    # Assuming Channel 1 is Mag-Heart
    radar_mag = radar_sample[1].cpu().numpy()
    axs[0].plot(t, radar_mag, 'b', alpha=0.6)
    axs[0].set_ylabel("Radar Mag (Input)")
    axs[0].set_title("Input: Mechanical Displacement")
    
    # 2. The Explainability Map (Heatmap)
    # We overlay the heatmap on the radar signal
    axs[1].plot(t, radar_mag, 'k', alpha=0.3)
    # Color the signal based on attention
    axs[1].scatter(t, radar_mag, c=cam_map, cmap='jet', s=5)
    axs[1].set_ylabel("Model Attention")
    axs[1].set_title("XAI: What the Model is Looking At (Red = High Attention)")
    
    # 3. ECG (Output)
    axs[2].plot(t, ecg_truth, 'g')
    axs[2].set_ylabel("ECG Lead II")
    axs[2].set_title("Output: Electrical R-Peak")
    
    # Draw lines connecting High Attention -> R-Peak
    # Find peaks in CAM
    from scipy.signal import find_peaks
    cam_peaks, _ = find_peaks(cam_map, height=0.5, distance=50)
    for p in cam_peaks:
        axs[1].axvline(x=p, color='r', linestyle='--', alpha=0.3)
        axs[2].axvline(x=p, color='r', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()