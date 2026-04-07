"""
Router Network with Curriculum Training (Time-Varying Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import configs.config as cfg

class RouterNetwork(nn.Module):
    """
    Learnable router (time-varying) with curriculum training support.
    This version uses 1D convolutions to produce a time-varying gate.
    """
    def __init__(self, in_channels=32, n_branches=2):
        super().__init__()
        self.n_branches = n_branches
        
        # --- MODIFICATION: Replaced AvgPool+Linear with Conv1d ---
        # This network slides along the time axis
        self.router_net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(0.1),       
            nn.Conv1d(64, n_branches, kernel_size=1) # Output [B, n_branches, T]
        )
        # --------------------------------------------------------
        
        # Initialize bias to favor first branch (stability)
        with torch.no_grad():
            self.router_net[-1].bias[0] = 0.7
            for i in range(1, n_branches):
                self.router_net[-1].bias[i] = 0.0
        
    def forward(self, x, epoch=0, total_epochs=1000):
        """
        Input: [B, C, T]
        Output: (gates [B, n_branches, T], entropy_loss [scalar])
        """
        
        # --- MODIFICATION: Logits are now time-varying ---
        logits = self.router_net(x)  # Shape is now [B, n_branches, T]
        # -------------------------------------------------

        # --- MODIFICATION: Apply softmax over branch dimension (dim=1) ---
        p = F.softmax(logits, dim=1)
        log_p = F.log_softmax(logits, dim=1)
        # Calculate entropy over branches, then average over time and batch
        entropy_loss = -torch.sum(p * log_p, dim=1).mean()
        # ---------------------------------------------------------------

        # Curriculum schedule
        mode, params = self.get_curriculum_params(epoch, total_epochs)

        if mode == 'forced_equal':
            gates = torch.ones_like(logits) / self.n_branches
        elif mode == 'annealed':
            temperature = params['temperature']
            # Apply gumbel_softmax on the branch dimension (dim=1)
            gates = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=1)
        else:  # mode == 'free'
            temperature = params['temperature']
            # Apply softmax on the branch dimension (dim=1)
            gates = F.softmax(logits / temperature, dim=1)

        # --- MODIFICATION: No need to expand, gates are already [B, n_branches, T] ---
        # The output shape is now what cast_ecg.py's fuse_normalized function expects.
        return gates, entropy_loss
    
    def get_curriculum_params(self, epoch, total_epochs):
        """Determine training mode based on absolute epoch numbers from cfg"""
        
        # Read the phase end-points from the config
        forced_end = getattr(cfg, 'curriculum_forced_equal', 40) # Use the default value if not found
        annealed_end = getattr(cfg, 'curriculum_annealed', 60)

        # Calculate the duration of the annealed phase
        annealed_duration = annealed_end - forced_end
        if annealed_duration <= 0:
            # Failsafe to prevent division by zero
            annealed_duration = 1 

        # --- Phase 1: Forced Equal ---
        if epoch < forced_end:
            return 'forced_equal', {}
        
        # --- Phase 2: Annealed ---
        elif epoch < annealed_end:
            # Calculate progress *within this phase* (a value from 0.0 to 1.0)
            progress = (epoch - forced_end) / annealed_duration
            
            # Cosine annealing from init temp down to final temp
            temp_init = 2.0
            temp_final = 0.5
            temperature = temp_final + 0.5 * (temp_init - temp_final) * (1 + np.cos(np.pi * progress))
            
            return 'annealed', {'temperature': temperature}
        # --- Phase 3: Free Routing ---
        else: 
            return 'free', {'temperature': 0.5}
        
    def get_gate_statistics(self, gates):
        """
        Compute statistics for monitoring router behavior
        gates: [B, n_branches, T]
        """
        
        # --- MODIFICATION: Update statistics for new shape ---
        # Entropy (higher = more diverse)
        # Average over batch and time
        entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=1).mean()
        
        # Dominance (lower = more balanced)
        # Average over batch and time
        dominance = gates.max(dim=1)[0].mean()
        
        # Branch utilization
        # Average over batch and time
        branch_usage = gates.mean(dim=(0, 2))  # [n_branches]
        # ------------------------------------------------------
        
        return {
            'entropy': entropy.item(),
            'dominance': dominance.item(),
            'branch_usage': branch_usage.cpu().numpy()
        }


# Test the module
if __name__ == '__main__':
    router = RouterNetwork(in_channels=32, n_branches=2)
    
    # Test at different epochs
    x = torch.randn(4, 32, 1024)
    
    print("Testing curriculum phases (TIME-VARYING ROUTER):")
    for epoch in [0, 50, 150, 250]:
        gates, entropy = router(x, epoch=epoch, total_epochs=250)
        stats = router.get_gate_statistics(gates)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Gates shape: {gates.shape}")
        print(f"  Entropy Loss: {entropy.item():.4f}")
        print(f"  Stat Entropy: {stats['entropy']:.4f}")
        print(f"  Stat Dominance: {stats['dominance']:.4f}")
        print(f"  Stat Branch usage: {stats['branch_usage']}")

    # Check that gates are actually time-varying
    gates_e150, _ = router(x, epoch=150, total_epochs=250)
    # Check std deviation across time
    time_std = gates_e150.std(dim=-1).mean()
    print(f"\nMean std dev across time: {time_std.item():.4f}")
    if time_std > 1e-4:
        print("✓ SUCCESS: Gates are time-varying.")
    else:
        print("✗ FAILED: Gates are still time-invariant.")
