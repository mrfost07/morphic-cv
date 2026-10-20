"""
Dynamic Sparse Training Engine (RigL - Rigged Lottery Ticket Algorithm).
Dynamically rewires the topology of sparse layers during training by:
1. Dropping the lowest magnitude active weights.
2. Growing new connections where the gradient signal is strongest.
Maintains a strict target sparsity (e.g., 80-90% zeros) from epoch 0 to convergence.
"""

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class RigLScheduler:
    """
    Manages dynamic topology evolution for targeted linear/conv layers in a model.
    """
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.8,
        update_frequency: int = 100,
        drop_fraction: float = 0.3,
        total_steps: int = 1000,
        exclude_layers: List[str] = None
    ):
        self.model = model
        self.target_sparsity = target_sparsity
        self.update_frequency = update_frequency
        self.initial_drop_fraction = drop_fraction
        self.total_steps = total_steps
        self.exclude_layers = exclude_layers or []
        
        self.step_count = 0
        self.masks: Dict[str, torch.Tensor] = {}
        self.modules: Dict[str, nn.Module] = {}
        
        # Initialize binary masks for eligible layers
        self._init_masks()

    def _init_masks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if any(ex in name for ex in self.exclude_layers):
                    continue
                w = module.weight.data
                # Initialize random sparse mask maintaining target_sparsity
                prob = torch.rand_like(w)
                k = int((1.0 - self.target_sparsity) * w.numel())
                if k < 1:
                    k = 1
                _, topk_idx = torch.topk(prob.view(-1), k=k)
                mask = torch.zeros_like(prob.view(-1))
                mask[topk_idx] = 1.0
                mask = mask.view_as(w)
                
                self.masks[name] = mask
                self.modules[name] = module
                # Apply initial mask to weights
                module.weight.data.mul_(mask)

    def apply_mask(self):
        """Zero out inactive weights in-place."""
        with torch.no_grad():
            for name, module in self.modules.items():
                mask = self.masks[name]
                module.weight.data.mul_(mask)

    def cosine_drop_fraction(self) -> float:
        """Anneal the drop fraction using cosine decay."""
        progress = min(1.0, self.step_count / max(1, self.total_steps))
        return self.initial_drop_fraction * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self):
        """
        Advance scheduler and rewire network topology if step matches update_frequency.
        Must be called after loss.backward() and before or after optimizer.step().
        """
        self.step_count += 1
        self.apply_mask()

        if self.step_count % self.update_frequency != 0:
            return

        current_drop_fraction = self.cosine_drop_fraction()
        if current_drop_fraction <= 0.001:
            return

        with torch.no_grad():
            for name, module in self.modules.items():
                mask = self.masks[name]
                w = module.weight.data
                grad = module.weight.grad
                if grad is None:
                    continue

                active_count = int(mask.sum().item())
                num_to_drop = int(active_count * current_drop_fraction)
                if num_to_drop < 1:
                    continue

                # 1. Prune: identify active weights with lowest absolute magnitude
                active_weights = torch.where(mask > 0.5, w.abs(), torch.tensor(float('inf'), device=w.device))
                _, drop_indices = torch.topk(active_weights.view(-1), k=num_to_drop, largest=False)

                new_mask = mask.clone().view(-1)
                new_mask[drop_indices] = 0.0

                # 2. Regrow: identify inactive connections with highest gradient magnitude
                inactive_grads = torch.where(new_mask < 0.5, grad.abs().view(-1), torch.tensor(-1.0, device=grad.device))
                _, grow_indices = torch.topk(inactive_grads, k=num_to_drop, largest=True)

                new_mask[grow_indices] = 1.0
                new_mask = new_mask.view_as(w)

                # Initialize newly grown weights to zero or small momentum
                self.masks[name] = new_mask
                module.weight.data.mul_(new_mask)

    def get_sparsity_stats(self) -> Dict[str, float]:
        """Returns the actual sparsity percentage per layer."""
        stats = {}
        total_params = 0
        total_zeros = 0
        for name, mask in self.masks.items():
            zeros = (mask == 0).sum().item()
            numel = mask.numel()
            stats[name] = zeros / numel
            total_params += numel
            total_zeros += zeros
        stats['overall'] = total_zeros / max(1, total_params)
        return stats
