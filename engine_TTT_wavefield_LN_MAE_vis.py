# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Test-Time Training Engine with Layer Normalization Adaptation for MAE
#
# This module implements the core training engine for Test-Time Training (TTT)
# using Masked Autoencoders with Vision Transformers. It provides online
# adaptation capabilities through LayerNorm parameter updates and self-supervised
# reconstruction losses.
#
# Key Features:
# - Online LayerNorm adaptation during test time
# - Masked autoencoder reconstruction for self-supervision  
# - Comprehensive visualization and logging capabilities
# - Support for ultrasound image classification tasks
#
# References:
# - DeiT: https://github.com/facebookresearch/deit
# - BEiT: https://github.com/microsoft/unilm/tree/master/beit
# - MAE: He, K., et al. "Masked autoencoders are scalable vision learners." CVPR 2022.
# --------------------------------------------------------

import math
import sys
import os.path
import copy
import glob
import argparse
from typing import Iterable, Optional, Dict, Any, List, Tuple, Union, Iterator
import logging

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Local imports
import models_mae_shared
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import timm.optim.optim_factory as optim_factory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@torch.no_grad()
@torch.no_grad()
def accuracy(labels_gt: torch.Tensor, pred_classes: torch.Tensor) -> float:
    """
    Compute classification accuracy between ground truth labels and predicted class indices.
    
    Args:
        labels_gt (torch.Tensor): Ground truth labels of shape (batch_size,) with integer class indices
        pred_classes (torch.Tensor): Predicted class indices of shape (batch_size,) with integer class indices
        
    Returns:
        float: Accuracy as percentage (0-100)
        
    Examples:
        >>> labels_gt = torch.tensor([0, 1, 0, 1])
        >>> pred_classes = torch.tensor([0, 1, 0, 0])  # One wrong prediction
        >>> accuracy(labels_gt, pred_classes)
        75.0
    """
    # Handle empty tensors
    if labels_gt.numel() == 0:
        return 0.0
    
    # Ensure tensors are on the same device
    if labels_gt.device != pred_classes.device:
        pred_classes = pred_classes.to(labels_gt.device)
    
    # Ensure correct data types
    labels_gt = labels_gt.long()
    pred_classes = pred_classes.long()
    
    # Calculate accuracy
    correct = (pred_classes == labels_gt).float()
    accuracy_pct = correct.mean() * 100.0
    
    return accuracy_pct.item()

class OnlineLayerNormUpdater:
    """
    Online Layer Normalization updater for test-time adaptation in Vision Transformers.
    
    This class provides adaptive capabilities for Vision Transformers by managing
    LayerNorm layers during test-time training. Unlike BatchNorm, LayerNorm normalizes
    across features rather than batches, making it more suitable for small batch scenarios
    common in test-time adaptation.
    
    Attributes:
        model (nn.Module): The Vision Transformer model containing LayerNorm layers
        momentum (float): Learning rate for parameter updates when enabled
        update_mode (str): Update strategy ('ema', 'fixed', or 'adaptive')
        ln_layers (List[Tuple[str, nn.LayerNorm]]): List of LayerNorm layers with names
        step_count (int): Number of adaptation steps performed
        
    Args:
        model (nn.Module): Vision Transformer model with LayerNorm layers
        momentum (float, optional): Learning rate for parameter updates. Defaults to 0.1
        update_mode (str, optional): Update strategy. Defaults to 'ema'
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        momentum: float = 0.1, 
        update_mode: str = 'ema'
    ):
        """Initialize the LayerNorm updater with the specified configuration."""
        self.model = model
        self.momentum = momentum
        self.update_mode = update_mode
        self.ln_layers: List[Tuple[str, nn.LayerNorm]] = []
        self.original_weights: List[Optional[torch.Tensor]] = []
        self.original_biases: List[Optional[torch.Tensor]] = []
        self.step_count = 0
        
        # Discover and store all LayerNorm layers
        self._find_ln_layers()
        self._store_original_params()
        
        # Log initialization details
        logger.info(f"LayerNorm Updater initialized:")
        logger.info(f"  - Found {len(self.ln_layers)} LayerNorm layers")
        logger.info(f"  - Update mode: {update_mode}")
        logger.info(f"  - Momentum: {momentum}")
        
        if self.ln_layers:
            logger.debug("LayerNorm layers found:")
            for name, _ in self.ln_layers[:5]:  # Show first 5
                logger.debug(f"  - {name}")
            if len(self.ln_layers) > 5:
                logger.debug(f"  ... and {len(self.ln_layers)-5} more")
        else:
            logger.warning("No LayerNorm layers found in the model!")
        
    def _find_ln_layers(self) -> None:
        """Discover and catalog all LayerNorm layers in the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.LayerNorm):
                self.ln_layers.append((name, module))
                
    def _store_original_params(self) -> None:
        """Store original LayerNorm parameters for potential reset operations."""
        for name, ln_layer in self.ln_layers:
            # Store weight parameters
            if ln_layer.weight is not None:
                self.original_weights.append(ln_layer.weight.clone().detach())
            else:
                self.original_weights.append(None)
                
            # Store bias parameters  
            if ln_layer.bias is not None:
                self.original_biases.append(ln_layer.bias.clone().detach())
            else:
                self.original_biases.append(None)
                
    def update_ln_online(
        self, 
        input_batch: torch.Tensor, 
        labels: Optional[torch.Tensor] = None, 
        adapt_params: bool = False
    ) -> None:
        """
        Perform online LayerNorm adaptation for test-time training.
        
        This method updates the model's LayerNorm behavior to adapt to the current
        test batch. It can either simply enable training mode for dynamic adaptation
        or perform gradient-based parameter updates.
        
        Args:
            input_batch (torch.Tensor): Current input batch for adaptation
            labels (Optional[torch.Tensor]): Optional labels (currently unused)
            adapt_params (bool): Whether to perform gradient-based parameter adaptation
        """
        self.step_count += 1
        
        # Store original training mode
        original_mode = self.model.training
        
        try:
            # Enable training mode for adaptation
            self.model.train()
            
            if adapt_params:
                # Perform gradient-based parameter adaptation
                self._adapt_ln_parameters(input_batch)
            else:
                # Simple adaptation: forward pass in training mode
                with torch.no_grad():
                    _ = self.model(input_batch, mask_ratio=0, reconstruct=False)
                    
        except Exception as e:
            logger.error(f"Error during LayerNorm adaptation: {e}")
            raise
        finally:
            # Restore original training mode
            self.model.train(original_mode)
        
    def _adapt_ln_parameters(self, input_batch: torch.Tensor) -> None:
        """
        Perform gradient-based adaptation of LayerNorm parameters.
        
        This method computes gradients for LayerNorm parameters and updates them
        using the specified momentum and update strategy.
        
        Args:
            input_batch (torch.Tensor): Input batch for computing adaptation loss
        """
        # Enable gradients for LayerNorm parameters
        for name, ln_layer in self.ln_layers:
            if ln_layer.weight is not None:
                ln_layer.weight.requires_grad_(True)
            if ln_layer.bias is not None:
                ln_layer.bias.requires_grad_(True)
        
        # Forward pass with gradient computation
        with torch.enable_grad():
            # Compute model outputs for adaptation loss
            outputs = self.model(input_batch, mask_ratio=0.0, reconstruct=False)
            
            # Compute adaptation loss (placeholder - can be customized)
            ln_loss = self._compute_adaptation_loss()
                
            # Compute gradients if loss requires them
            if ln_loss.requires_grad:
                ln_params = [
                    p for name, module in self.ln_layers 
                    for p in [module.weight, module.bias] 
                    if p is not None and p.requires_grad
                ]
                
                if ln_params:
                    ln_grads = torch.autograd.grad(
                        ln_loss, 
                        ln_params,
                        retain_graph=False, 
                        allow_unused=True
                    )
                    
                    # Apply parameter updates
                    self._update_ln_parameters(ln_grads)
    
    def _compute_adaptation_loss(self) -> torch.Tensor:
        """
        Compute adaptation loss for LayerNorm parameter updates.
        
        This is a placeholder implementation that computes a simple regularization
        loss. In practice, this could be replaced with more sophisticated losses
        such as reconstruction loss or feature consistency loss.
        
        Returns:
            torch.Tensor: Computed adaptation loss
        """
        ln_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for name, ln_layer in self.ln_layers:
            if ln_layer.weight is not None:
                # Simple regularization: encourage weights to stay close to mean
                weight_reg = torch.norm(ln_layer.weight - ln_layer.weight.mean()) * 0.01
                ln_loss = ln_loss + weight_reg
                
        return ln_loss
    
    def _update_ln_parameters(self, gradients: List[torch.Tensor]) -> None:
        """
        Update LayerNorm parameters using computed gradients.
        
        Args:
            gradients (List[torch.Tensor]): List of gradients for LayerNorm parameters
        """
        param_idx = 0
        
        for name, ln_layer in self.ln_layers:
            # Update weight parameters
            if ln_layer.weight is not None and param_idx < len(gradients):
                if gradients[param_idx] is not None:
                    update = gradients[param_idx] * self.momentum
                    ln_layer.weight.data.subtract_(update)
                param_idx += 1
                
            # Update bias parameters
            if ln_layer.bias is not None and param_idx < len(gradients):
                if gradients[param_idx] is not None:
                    update = gradients[param_idx] * self.momentum
                    ln_layer.bias.data.subtract_(update)
                param_idx += 1
    
    def update_bn_online(
        self, 
        input_batch: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> None:
        """
        Wrapper method for backward compatibility with BatchNorm interfaces.
        
        Args:
            input_batch (torch.Tensor): Input batch for adaptation
            labels (Optional[torch.Tensor]): Optional labels (unused)
        """
        return self.update_ln_online(input_batch, labels, adapt_params=False)
    
    def get_bn_statistics(self) -> Dict[str, Any]:
        """
        Retrieve current LayerNorm statistics for monitoring and analysis.
        
        Note: Unlike BatchNorm, LayerNorm doesn't maintain running statistics,
        so we return the current parameters and configuration.
        
        Returns:
            Dict[str, Any]: Dictionary containing LayerNorm statistics
        """
        if not self.ln_layers:
            return {'step_count': self.step_count, 'num_layers': 0}
            
        stats = {}
        for name, ln_layer in self.ln_layers:
            try:
                layer_stats = {
                    'weight': ln_layer.weight.clone().detach() if ln_layer.weight is not None else None,
                    'bias': ln_layer.bias.clone().detach() if ln_layer.bias is not None else None,
                    'normalized_shape': ln_layer.normalized_shape,
                    'eps': ln_layer.eps,
                    'elementwise_affine': ln_layer.elementwise_affine
                }
                stats[name] = layer_stats
            except Exception as e:
                logger.error(f"Error collecting statistics for {name}: {e}")
                
        stats['meta'] = {
            'step_count': self.step_count,
            'num_layers': len(self.ln_layers),
            'update_mode': self.update_mode,
            'momentum': self.momentum
        }
        
        return stats
    
    def reset_ln_parameters(self) -> None:
        """Reset LayerNorm parameters to their original initialization values."""
        for i, (name, ln_layer) in enumerate(self.ln_layers):
            try:
                # Reset weight parameters
                if (i < len(self.original_weights) and 
                    self.original_weights[i] is not None and 
                    ln_layer.weight is not None):
                    ln_layer.weight.data.copy_(self.original_weights[i])
                    
                # Reset bias parameters
                if (i < len(self.original_biases) and 
                    self.original_biases[i] is not None and 
                    ln_layer.bias is not None):
                    ln_layer.bias.data.copy_(self.original_biases[i])
                    
            except Exception as e:
                logger.error(f"Error resetting parameters for layer {name}: {e}")
        
        self.step_count = 0
        logger.info("LayerNorm parameters reset to original values")


class SimpleViTAdaptation:
    """
    Simplified test-time adaptation strategy for Vision Transformers.
    
    This class provides a lightweight adaptation approach that focuses on
    enabling training mode during test time to activate stochastic components
    like dropout while avoiding complex parameter updates.
    
    Attributes:
        model (nn.Module): The Vision Transformer model to adapt
        step_count (int): Number of adaptation steps performed
        has_dropout (bool): Whether the model contains dropout layers
        has_layernorm (bool): Whether the model contains LayerNorm layers
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize the simple ViT adaptation strategy.
        
        Args:
            model (nn.Module): Vision Transformer model to adapt
        """
        self.model = model
        self.step_count = 0
        
        # Analyze model components for adaptation
        self.has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
        self.has_layernorm = any(isinstance(m, torch.nn.LayerNorm) for m in model.modules())
        
        logger.info("Simple ViT Adaptation initialized:")
        logger.info(f"  - Has Dropout layers: {self.has_dropout}")
        logger.info(f"  - Has LayerNorm layers: {self.has_layernorm}")
        
    def update_bn_online(
        self, 
        input_batch: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> None:
        """
        Perform simple test-time adaptation through training mode activation.
        
        This method enables training mode temporarily to activate stochastic
        behaviors like dropout, providing a form of test-time regularization.
        
        Args:
            input_batch (torch.Tensor): Current input batch
            labels (Optional[torch.Tensor]): Optional labels (unused)
        """
        self.step_count += 1
        
        # Store original training mode
        original_mode = self.model.training
        
        try:
            # Enable training mode for stochastic behavior
            self.model.train()
            
            # Forward pass to enable training-mode behaviors
            with torch.no_grad():
                _ = self.model(input_batch, mask_ratio=0, reconstruct=False)
                
        except Exception as e:
            logger.error(f"Error during simple ViT adaptation: {e}")
        finally:
            # Restore original training mode
            self.model.train(original_mode)
        
    def get_bn_statistics(self) -> Dict[str, Any]:
        """
        Return adaptation statistics and model state information.
        
        Returns:
            Dict[str, Any]: Dictionary containing adaptation statistics
        """
        return {
            'step_count': self.step_count,
            'has_dropout': self.has_dropout,
            'has_layernorm': self.has_layernorm,
            'model_training_mode': self.model.training,
            'adaptation_type': 'simple_vit'
        }

class OnlineBatchNormUpdater:
    """
    Online Batch Normalization updater for test-time adaptation
    """
    
    def __init__(self, model, momentum=0.1, update_mode='exponential'):
        """
        Initialize the online BN updater
        
        Args:
            model: Neural network model with BN layers
            momentum: Update momentum for running statistics
            update_mode: 'exponential' or 'cumulative' update strategy
        """
        self.model = model
        self.momentum = momentum
        self.update_mode = update_mode
        self.bn_layers = []
        self.original_momentums = []
        self.step_count = 0
        
        # Find and store all BN layers
        self._find_bn_layers()
        
    def _find_bn_layers(self):
        """Find all batch normalization layers in the model"""
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                self.bn_layers.append((name, module))
                self.original_momentums.append(module.momentum)
                
    def set_bn_momentum(self, momentum):
        """Set momentum for all BN layers"""
        for _, bn_layer in self.bn_layers:
            bn_layer.momentum = momentum
            
    def reset_bn_momentum(self):
        """Reset BN momentum to original values"""
        for i, (_, bn_layer) in enumerate(self.bn_layers):
            bn_layer.momentum = self.original_momentums[i]
            
    def update_bn_online(self, input_batch, labels=None):
        """
        Update BN statistics online with current batch
        
        Args:
            input_batch: Current input batch
            labels: Optional labels (not used for BN update but can be logged)
        """
        self.step_count += 1
        
        # Set model to training mode to update BN statistics
        original_mode = self.model.training
        self.model.train()
        
        # Adjust momentum based on update strategy
        if self.update_mode == 'cumulative':
            # Use 1/n momentum for cumulative average
            adaptive_momentum = 1.0 / self.step_count
            self.set_bn_momentum(adaptive_momentum)
        elif self.update_mode == 'exponential':
            # Use fixed exponential momentum
            self.set_bn_momentum(self.momentum)
        elif self.update_mode == 'adaptive':
            # Adaptive momentum that decreases over time
            adaptive_momentum = max(0.01, self.momentum / (1 + 0.1 * self.step_count))
            self.set_bn_momentum(adaptive_momentum)
            
        # Forward pass to update BN running statistics
        with torch.no_grad():
            _ = self.model(input_batch, mask_ratio=0, reconstruct=False)
            
        # Restore original training mode
        self.model.train(original_mode)
        
    def get_bn_statistics(self):
        """Get current BN statistics for monitoring"""
        stats = {}
        for name, bn_layer in self.bn_layers:
            stats[name] = {
                'running_mean': bn_layer.running_mean.clone(),
                'running_var': bn_layer.running_var.clone(),
                'momentum': bn_layer.momentum
            }
        return stats
        
    def reset_bn_statistics(self):
        """Reset BN running statistics"""
        for _, bn_layer in self.bn_layers:
            bn_layer.reset_running_stats()
        self.step_count = 0


# Enhanced version with accuracy metrics
def plot_prediction_comparison_with_metrics(labels_gt, labels_pred_init, labels_pred_TTT, sampID, save_path=None):
    """
    Enhanced version with improved ground truth visibility
    """
    plt.figure(figsize=(14, 8))
    
    # Create frame indices
    frame_indices = range(len(labels_gt))
    
    # Plot the three curves with improved visibility for ground truth
    # Ground truth with thicker line, larger markers, and distinctive style
    plt.plot(frame_indices, labels_gt, 'o-', linewidth=4, markersize=8, 
             color='black', alpha=0.9, label='Ground Truth', markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='black', zorder=3)
    
    # Initial predictions with dashed line
    plt.plot(frame_indices, labels_pred_init, 's--', linewidth=2, markersize=5, 
             color='red', alpha=0.8, label='Initial Pred.', zorder=1)
    
    # TTT predictions with dotted line
    plt.plot(frame_indices, labels_pred_TTT, '^:', linewidth=2, markersize=5, 
             color='blue', alpha=0.8, label='Improved Pred. by TTT', zorder=2)
    
    # Calculate accuracies
    acc_init = (np.array(labels_gt) == np.array(labels_pred_init)).sum() / len(labels_gt)*100
    acc_ttt = (np.array(labels_gt) == np.array(labels_pred_TTT)).sum() / len(labels_gt)*100
    improvement = acc_ttt - acc_init
    
    # Customize the plot
    plt.xlabel('Frame Index (#)', fontsize=12, fontweight='bold')
    plt.ylabel('Condition Label (OK/NG)', fontsize=12, fontweight='bold')
    plt.title(f'Prediction Comparison for Sample-{sampID}\n'
              f'Initial Acc: {acc_init:.2f}% | TTT Acc: {acc_ttt:.2f}% | '
              f'Improvement: {improvement:+.2f}%', fontsize=14, fontweight='bold')
    
    # Set y-axis to show discrete labels with more spacing
    plt.yticks([0, 1], ['OK (0)', 'NG (1)'], fontsize=11, fontweight='bold')
    plt.ylim(-0.2, 1.2)
    
    # Add horizontal reference lines for each class
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    plt.axhline(y=1, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend with improved formatting
    legend_labels = [
        'Ground Truth (Reference)',
        f'Initial Pred. (Acc: {acc_init:.1f}%)',
        f'TTT Improved Pred. (Acc: {acc_ttt:.1f}%)'
    ]
    
    handles, _ = plt.gca().get_legend_handles_labels()
    plt.legend(handles, legend_labels, loc='upper left', 
              frameon=True, fancybox=True, shadow=True, fontsize=11,
              bbox_to_anchor=(0.02, 0.995))
    
    # Highlight correct/incorrect predictions with background shading
    for i, (gt, init, ttt) in enumerate(zip(labels_gt, labels_pred_init, labels_pred_TTT)):
        if gt != init and gt == ttt:  # TTT corrected (green background)
            plt.axvspan(i-0.4, i+0.4, color='green', alpha=0.15, zorder=0)
        elif gt == init and gt != ttt:  # TTT made it worse (red background)
            plt.axvspan(i-0.4, i+0.4, color='red', alpha=0.15, zorder=0)
        elif gt != init and gt != ttt:  # Both wrong (light gray background)
            plt.axvspan(i-0.4, i+0.4, color='gray', alpha=0.1, zorder=0)
    
    # Add text annotations for key statistics
    stats_text = (f'Frames: {len(labels_gt)} | '
                 f'GT=Init={len([1 for g,i in zip(labels_gt, labels_pred_init) if g==i])} | '
                 f'GT=TTT={len([1 for g,t in zip(labels_gt, labels_pred_TTT) if g==t])}')
    
    plt.figtext(0.5, 0.12, stats_text, ha='center', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for the bottom text
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced prediction comparison plot saved to: {save_path}")
    
    plt.show()
    plt.close()

def plot_ttt_results(all_results, sampID, save_path=None, plotOptions='accuracy'):
    """
    Plot TTT-MAE training results with transposed data
    
    Args:
        all_results: List of lists [steps_per_example][samples], needs to be transposed
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Transpose all_results: from [steps][samples] to [samples][steps]
    all_results_transposed = np.array(all_results).T
    
    # Plot each curve (each sample/data point)
    for sample_idx in range(all_results_transposed.shape[0]):
        sample_results = all_results_transposed[sample_idx]
        epochs = range(len(sample_results))
        plt.plot(epochs, sample_results, 
                marker='o', 
                linewidth=1.5, 
                markersize=3,
                label=f'{sample_idx}',  # Just the index number
                alpha=0.7)
    
    # Customize the plot
    plt.xlabel('TTT-MAE Epochs', fontsize=12, fontweight='bold')
    
    if plotOptions == 'accuracy':
        plt.ylabel('Frame Accuracy', fontsize=12, fontweight='bold')
        plt.title(f"Test-Time Training Performance for Sample-{sampID}", fontsize=14, fontweight='bold')
    else:
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title(f"Test-Time Training Loss for Sample-{sampID}", fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend with just indices
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              frameon=True, fancybox=True, shadow=True,
              ncol=1, fontsize=8)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    plt.close()
    
    

    
# ...existing code...
def visualize_latents_with_hyperplane(model, dataloader, device, save_path=None, num_classes=2,n_samples=None, reduce_method='pca', n_components=2, class_names=None, verbose=False):
    """
    Improved feature extraction & classifier-weight handling tailored to MaskedAutoencoderViT
    - Prefer calling model.forward_encoder / forward_features to get latent[:,0] directly (no hooks).
    - If head_type == 'vit_head' map classifier_pred weights back to encoder latent space
      via classifier_embed linear layer to get an interpretable decision vector.
    - Fall back to a robust hook if neither API exists.
    """
    import math
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np

    model_cpu_state = model.training
    model.eval()
    model.to(device)

    # collect features and labels
    features_list = []
    labels_list = []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                samples, labels = batch[0], batch[1]
            else:
                samples = batch
                labels = None
            samples = samples.to(device)

            # Preferred: use encoder / feature API from MaskedAutoencoderViT
            feat_tensor = None
            try:
                if hasattr(model, 'forward_encoder'):
                    # forward_encoder returns (latent, mask, ids_restore); latent shape (B, L, D)
                    latent, _, _ = model.forward_encoder(samples, mask_ratio=0)
                    feat_tensor = latent[:, 0].detach().cpu()  # cls token feature (B, D)
                elif hasattr(model, 'forward_features'):
                    out = model.forward_features(samples)  # may return tokens
                    # try to get cls token or pooled vector
                    if isinstance(out, tuple):
                        out = out[0]
                    if out.ndim == 3:
                        feat_tensor = out[:, 0].detach().cpu()
                    else:
                        feat_tensor = out.detach().cpu()
                else:
                    # fallback: call model.forward and use the returned latent (third returned item)
                    out = model(samples, None, mask_ratio=0, reconstruct=False)
                    # model.forward returns (loss_dict, pred, latent[:,0], head)
                    if isinstance(out, (list, tuple)) and len(out) >= 3:
                        feat_tensor = out[2].detach().cpu()
            except Exception:
                feat_tensor = None

            # Robust hook fallback if APIs above failed
            if feat_tensor is None:
                # try a small hook on last transformer block or classifier inputs
                hook_feat = {}
                def _hook_fn(module, inp, outp):
                    try:
                        if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                            hook_feat['feat'] = inp[0].detach().cpu()
                        elif isinstance(outp, torch.Tensor):
                            hook_feat['feat'] = outp.detach().cpu()
                    except Exception:
                        hook_feat['feat'] = None

                # choose fallback target: prefer classifier_embed or last block
                fallback_target = None
                if hasattr(model, 'classifier_embed'):
                    fallback_target = model.classifier_embed
                else:
                    for name, mod in model.named_modules():
                        if 'block' in name or 'encoder' in name or 'transformer' in name:
                            fallback_target = mod
                    if fallback_target is None:
                        # last linear as final fallback
                        for name, mod in model.named_modules():
                            if isinstance(mod, torch.nn.Linear):
                                fallback_target = mod

                h = None
                try:
                    if fallback_target is not None:
                        h = fallback_target.register_forward_hook(_hook_fn)
                        # repeat forward attempts
                        try:
                            _ = model.forward_encoder(samples, mask_ratio=0) if hasattr(model, 'forward_encoder') else None
                        except Exception:
                            try:
                                _ = model(samples, None, mask_ratio=0, reconstruct=False)
                            except Exception:
                                _ = model(samples)
                        feat_tensor = hook_feat.get('feat', None)
                finally:
                    if h is not None:
                        try:
                            h.remove()
                        except Exception:
                            pass

            if feat_tensor is None:
                raise RuntimeError("Failed to extract features for visualization. Inspect model.named_modules() to pick hook target.")

            # feat_tensor expected shape (B, D)
            if feat_tensor.ndim == 3:
                # collapse sequence dimension: take cls token
                feat_tensor = feat_tensor[:, 0, :]

            feat_np = feat_tensor.cpu().numpy()
            features_list.append(feat_np)
            if labels is not None:
                labels_list.append(labels.detach().cpu().numpy())
            else:
                labels_list.append(np.zeros((feat_np.shape[0],), dtype=int))

            count += feat_np.shape[0]
            if n_samples is not None and count >= n_samples:
                break

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    if n_samples is not None and X.shape[0] > n_samples:
        X = X[:n_samples]; y = y[:n_samples]

    if verbose:
        print(f"Collected features X.shape={X.shape}, y.shape={y.shape}")

    # PCA reduction
    if reduce_method.lower() == 'pca':
        pca = PCA(n_components=n_components)
        X2 = pca.fit_transform(X)
    else:
        raise NotImplementedError("Only 'pca' reduction implemented.")

    # Obtain classifier weight in encoder latent space
    weight = None
    bias = None
    try:
        if getattr(model, 'head_type', None) == 'vit_head' and hasattr(model, 'classifier_pred') and hasattr(model, 'classifier_embed'):
            # classifier_pred: (out_features, classifier_embed_dim)
            W_pred = model.classifier_pred.weight.detach().cpu().numpy()  # (out, C_e)
            b_pred = model.classifier_pred.bias.detach().cpu().numpy() if model.classifier_pred.bias is not None else np.zeros(W_pred.shape[0])
            W_embed = model.classifier_embed.weight.detach().cpu().numpy()  # (C_e, D)
            b_embed = model.classifier_embed.bias.detach().cpu().numpy() if model.classifier_embed.bias is not None else np.zeros(W_embed.shape[0])
            # map back to encoder latent space: W_combined = W_pred @ W_embed  -> shape (out, D)
            W_combined = W_pred.dot(W_embed)
            b_combined = b_pred + W_pred.dot(b_embed)
            w = W_combined
            b = b_combined
        elif hasattr(model, 'head') and getattr(model, 'head_type', None) == 'linear':
            # linear head after bn (bn.affine=False here in model), so direct mapping
            w = model.head.weight.detach().cpu().numpy()  # (out, D)
            b = model.head.bias.detach().cpu().numpy() if model.head.bias is not None else np.zeros(w.shape[0])
        else:
            # generic fallback: find last Linear layer weights
            found = None
            for name, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    found = mod
            if found is None:
                raise RuntimeError("No Linear classifier found for weight extraction.")
            w = found.weight.detach().cpu().numpy()
            b = found.bias.detach().cpu().numpy() if found.bias is not None else np.zeros(w.shape[0])

        # derive single decision vector for binary case
        if w.shape[0] == 1 or w.shape[0] == num_classes:
            if w.shape[0] == 1:
                weight = w[0]; bias = b[0]
            elif w.shape[0] == 2:
                weight = (w[1] - w[0]); bias = (b[1] - b[0])
            else:
                weight = (w[1] - w[0]); bias = (b[1] - b[0])
        else:
            weight = w[0]; bias = b[0]
    except Exception as e:
        weight = None; bias = None
        if verbose:
            print("Could not extract classifier weights:", e)

    # Project decision vector into PCA space: compute decision values on PCA grid
    xx = yy = Z = None
    if weight is not None:
        margin = 0.5
        x_min, x_max = X2[:,0].min() - margin, X2[:,0].max() + margin
        y_min, y_max = X2[:,1].min() - margin, X2[:,1].max() + margin
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (G,2)
        components = pca.components_  # (2, D)
        mean = pca.mean_              # (D,)
        X_approx = grid.dot(components) + mean  # (G, D)
        decision_vals = X_approx.dot(weight) + bias
        Z = decision_vals.reshape(xx.shape)

    # Z = 1-Z  # Invert decision values for correct contour orientation
    
    # Plot
    plt.figure(figsize=(8,6))
    colors = ['tab:blue', 'tab:red']
    # colors = ['tab:red', 'tab:blue']
    
    for lab in np.unique(y):
        mask = (y == lab)
        plt.scatter(X2[mask,0], X2[mask,1], s=30, alpha=0.8,
                    label=(class_names[lab] if class_names is not None else f'class_{lab}'),
                    color=colors[int(lab) % len(colors)], edgecolor='k')
    if Z is not None:
        plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        # plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.15)
        plt.contourf(xx, yy, Z, levels=40, cmap='coolwarm', alpha=0.15)        

    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.title('Latent features (2D PCA) and classifier hyperplane')
    plt.legend(); plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved latent visualization to {save_path}")
    plt.show()
    plt.close()

    model.train(model_cpu_state)
    return X, X2, y
    
def get_prameters_from_args(model, args):
    if args.finetune_mode == 'encoder':
        for name, p in model.named_parameters():
            if name.startswith('decoder'):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == 'all':
        parameters = model.parameters()
    elif args.finetune_mode == 'encoder_no_cls_no_msk':
        for name, p in model.named_parameters():
            if name.startswith('decoder') or name == 'cls_token' or name == 'mask_token':
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    return parameters


def _reinitialize_model(base_model, base_optimizer, base_scalar, clone_model, args, device):
    if args.stored_latents:
        # We don't need to change the model, as it is never changed
        base_model.train(True)
        base_model.to(device)
        return base_model, base_optimizer, base_scalar
    clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()), strict=False)
    clone_model.train(True)
    clone_model.to(device)
    if args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(get_prameters_from_args(clone_model, args), lr=args.lr, momentum=args.optimizer_momentum)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    else:
        assert args.optimizer_type == 'adam_w'
        optimizer = torch.optim.AdamW(get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95))
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    if args.load_loss_scalar:
        loss_scaler.load_state_dict(base_scalar.state_dict())
    return clone_model, optimizer, loss_scaler

class TTTViTAdaptation:
    """Actual test-time training with gradient updates"""
    
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.lr = lr
        self.step_count = 0
        
        # Create optimizer for adaptation
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def update_bn_online(self, input_batch, labels=None):
        """Real test-time training with gradient updates"""
        self.step_count += 1
        
        self.model.train()
        
        # Self-supervised adaptation using reconstruction
        self.optimizer.zero_grad()
        
        with torch.enable_grad():
            # Use reconstruction loss for adaptation
            loss_dict, _, _, _ = self.model(input_batch, None, mask_ratio=0.75)
            
            if 'reconstruction' in loss_dict:
                adapt_loss = loss_dict['reconstruction']
            elif isinstance(loss_dict, dict):
                adapt_loss = sum(loss_dict.values())  # Sum all losses
            else:
                adapt_loss = loss_dict
                
            # Gradient step
            adapt_loss.backward()
            self.optimizer.step()
            
            # print(f"TTT adaptation loss: {adapt_loss.item():.4f}")
        
    def get_bn_statistics(self):
        return {
            'step_count': self.step_count,
            'lr': self.lr,
            'model_mode': 'training'
        }
        
def _get_classifier_config(model_name: str) -> Tuple[int, int, int]:
    """
    Get classifier configuration parameters based on model name.
    
    Args:
        model_name: Name of the MAE ViT model
        
    Returns:
        Tuple of (depth, embed_dim, num_heads)
        
    Raises:
        AssertionError: If model_name is not supported
    """
    config_map = {
        'mae_vit_small_patch16': (8, 512, 16),
        'mae_vit_base_patch16': (8, 768, 12),
        'mae_vit_huge_patch14': (12, 768, 12),
        'mae_vit_large_patch16': (12, 768, 12)
    }
    
    if model_name in config_map:
        return config_map[model_name]
    
    # Handle partial matches for huge and large models
    for key in config_map:
        if key in model_name:
            return config_map[key]
    
    raise AssertionError(f"Unsupported model: {model_name}")


def _setup_data_loaders(dataset_train: torch.utils.data.Dataset,
                       dataset_val: torch.utils.data.Dataset,
                       batch_size: int,
                       num_workers: int) -> Tuple[Iterator, Iterator, Iterator]:
    """
    Setup data loaders for training and validation.
    
    Args:
        dataset_train: Training dataset
        dataset_val: Validation dataset  
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, vis_loader)
    """
    train_loader = iter(torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    ))
    
    val_loader = iter(torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    ))
    
    vis_loader = iter(torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    ))
    
    return train_loader, val_loader, vis_loader


def _initialize_clone_model(args: argparse.Namespace,
                          num_classes: int) -> torch.nn.Module:
    """
    Initialize a clone of the MAE model with specified configuration.
    
    Args:
        args: Parsed command line arguments
        num_classes: Number of output classes
        
    Returns:
        Initialized clone model
    """
    classifier_depth, classifier_embed_dim, classifier_num_heads = _get_classifier_config(args.model)
    
    clone_model = models_mae_shared.__dict__[args.model](
        num_classes=num_classes,
        head_type=args.head_type,
        norm_pix_loss=args.norm_pix_loss,
        classifier_depth=classifier_depth,
        classifier_embed_dim=classifier_embed_dim,
        classifier_num_heads=classifier_num_heads,
        rotation_prediction=False
    )
    
    return clone_model


def _setup_adaptation_strategy(model: torch.nn.Module, 
                             args: argparse.Namespace) -> Any:
    """
    Setup the adaptation strategy for test-time training.
    
    Args:
        model: The neural network model
        args: Parsed command line arguments
        
    Returns:
        Configured adaptation updater
    """
    ln_momentum = getattr(args, 'ln_momentum', 0.1)
    ln_update_mode = getattr(args, 'ln_update_mode', 'ema')
    adapt_ln_params = getattr(args, 'adapt_ln_params', False)
    
    if adapt_ln_params:
        online_updater = OnlineLayerNormUpdater(
            model, 
            momentum=ln_momentum, 
            update_mode=ln_update_mode
        )
        print("Using LayerNorm parameter adaptation")
    else:
        online_updater = SimpleViTAdaptation(model)
        print("Using simple ViT adaptation (training mode only)")
    
    return online_updater


def _perform_test_time_training(model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer,
                              loss_scaler: Any,
                              samples: torch.Tensor,
                              test_samples: torch.Tensor,
                              test_label: torch.Tensor,
                              args: argparse.Namespace,
                              data_iter_step: int,
                              metric_logger: misc.MetricLogger) -> Tuple[List[np.ndarray], List[float]]:
    """
    Perform test-time training for a single batch.
    
    Args:
        model: Neural network model
        optimizer: Model optimizer
        loss_scaler: Loss scaling utility
        samples: Training samples
        test_samples: Test samples for evaluation
        test_label: Test labels
        args: Parsed command line arguments
        data_iter_step: Current data iteration step
        metric_logger: Metrics logging utility
        
    Returns:
        Tuple of (all_predictions, all_accuracies)
    """
    accum_iter = args.accum_iter
    all_predictions = []
    all_accuracies = []
    
    # Test time training loop
    for step_per_example in range(args.steps_per_example * accum_iter):
        mask_ratio = args.mask_ratio
        
        # Training phase with MAE reconstruction
        model.train()
        
        loss_dict, _, _, _ = model(samples, None, mask_ratio=mask_ratio)
        loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
        loss_value = loss.item()
        loss /= accum_iter
        
        if not math.isfinite(loss_value):
            logging.error(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
            
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                   update_grad=(step_per_example + 1) % accum_iter == 0)
        
        if (step_per_example + 1) % accum_iter == 0:
            if args.verbose:
                logging.info(f'Datapoint {data_iter_step} iter {step_per_example}: rec_loss {loss_value}')
            
            optimizer.zero_grad()
                
        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Evaluation phase
        if (step_per_example + 1) % accum_iter == 0:
            model.eval()
            with torch.no_grad():
                batch_predictions = []
                for inference_iter in range(accum_iter):
                    loss_d, _, _, pred = model(test_samples, test_label, mask_ratio=0, reconstruct=False)
                    
                    cls_loss = loss_d['classification'].item()

                    if args.verbose:
                        logging.info(f'Datapoint {data_iter_step} iter {step_per_example}: class_loss {cls_loss}')
                        
                    batch_predictions.append(pred.argmax(axis=1).detach().cpu().numpy())
                
                # Calculate accuracy
                acc1 = accuracy(test_label.cpu().detach(), torch.tensor(np.array(batch_predictions[0])))
                
                if (step_per_example + 1) // accum_iter == args.steps_per_example:
                    metric_logger.update(top1_acc=acc1)
                    metric_logger.update(loss=cls_loss)
                
                all_predictions.append(batch_predictions[0])
                all_accuracies.append(acc1)
    
    return all_predictions, all_accuracies


def train_on_test_with_online_ln_update_MAE(base_model: torch.nn.Module,
                                          base_optimizer: torch.optim.Optimizer,
                                          base_scalar: Any,
                                          dataset_train: torch.utils.data.Dataset,
                                          dataset_val: torch.utils.data.Dataset,
                                          device: torch.device,
                                          log_writer: Optional[Any] = None,
                                          args: Optional[argparse.Namespace] = None,
                                          num_classes: int = 1000,
                                          iter_start: int = 0) -> float:
    """
    Perform test-time training with online LayerNorm adaptation for MAE models.
    
    This function implements Test-Time Training (TTT) using Masked Autoencoders (MAE)
    with online LayerNorm parameter adaptation. The model adapts to test data through
    self-supervised reconstruction tasks while maintaining classification performance.
    
    Args:
        base_model: Pre-trained base model for initialization
        base_optimizer: Base optimizer configuration
        base_scalar: Base loss scaler configuration
        dataset_train: Training dataset (for reference)
        dataset_val: Validation/test dataset for adaptation
        device: Computing device (CPU/GPU)
        log_writer: Optional logging writer
        args: Parsed command line arguments containing hyperparameters
        num_classes: Number of classification classes
        iter_start: Starting iteration number
        
    Returns:
        Final test-time training accuracy
        
    Raises:
        SystemExit: If training loss becomes non-finite
    """
    if args is None:
        raise ValueError("Arguments cannot be None")
    
    logging.info(f"Starting test-time training for sample {args.sampID}")
    
    # Initialize model configuration
    clone_model = _initialize_clone_model(args, num_classes)
    
    # Setup metric logger and data loaders
    metric_logger = misc.MetricLogger(delimiter="  ")
    batch_wavefields = args.batch_size
    
    train_loader, val_loader, vis_dataloader_init = _setup_data_loaders(
        dataset_train, dataset_val, batch_wavefields, args.num_workers
    )
    
    # Setup visualization path
    save_path_pre = os.path.join(args.output_dir, 'latent_vis_TTT_Init.png')
    
    # Reinitialize model with base parameters
    model, optimizer, loss_scaler = _reinitialize_model(
        base_model, base_optimizer, base_scalar, clone_model, args, device
    )
    
    # Setup adaptation strategy
    online_updater = _setup_adaptation_strategy(model, args)
    
    # Initialize tracking variables
    dataset_len = len(dataset_val)
    num_itr = int(np.floor(dataset_len / batch_wavefields))
    accum_iter = args.accum_iter
    
    all_results = torch.zeros((num_itr, args.steps_per_example), dtype=torch.float32)
    all_losses = torch.zeros((num_itr, args.steps_per_example), dtype=torch.float32)
    
    # Prediction tracking
    labels_gt = []
    labels_pred_init = []
    labels_pred_TTT = []
    bn_stats_history = []
    
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    logging.info(f"Starting TTT with {num_itr} iterations, {args.steps_per_example} steps per example")
    
    # Main test-time training loop
    for data_iter_step in range(iter_start, num_itr):
        # Get next validation batch
        val_data = next(val_loader)
        test_samples, test_label = val_data
        test_samples = test_samples.to(device, non_blocking=True)
        samples = test_samples.clone()
        test_label = test_label.to(device, non_blocking=True)
        
        # Track ground truth labels
        labels_gt.extend(test_label.cpu().detach().numpy().tolist())
        
        # Log progress
        if data_iter_step == 0:
            logging.info(f"LayerNorm adaptation for test sample-{args.sampID} batch-{data_iter_step}")
        
        print(f"=>{data_iter_step}", end="")
        
        # Perform test-time training
        all_predictions, all_accuracies = _perform_test_time_training(
            model, optimizer, loss_scaler, samples, test_samples, test_label,
            args, data_iter_step, metric_logger
        )
        
        # Store predictions for analysis
        if len(all_predictions) > 0:
            if data_iter_step == 0:  # First step predictions
                labels_pred_init.extend(all_predictions[0].tolist())
            
            # Last step predictions  
            labels_pred_TTT.extend(all_predictions[-1].tolist())
        
        # Store results
        for idx, acc in enumerate(all_accuracies):
            if idx < args.steps_per_example:
                all_results[data_iter_step, idx] = acc
    
    # Calculate final accuracies
    acc_samp_init = accuracy(
        torch.tensor(np.array(labels_gt)), 
        torch.tensor(np.array(labels_pred_init))
    )
    acc_samp_TTT = accuracy(
        torch.tensor(np.array(labels_gt)), 
        torch.tensor(np.array(labels_pred_TTT))
    )
    
    hit_cnt_TTT = (torch.tensor(np.array(labels_gt)) == torch.tensor(np.array(labels_pred_TTT))).sum().item()
    hit_cnt_init = (torch.tensor(np.array(labels_gt)) == torch.tensor(np.array(labels_pred_init))).sum().item()
    num_samples = len(labels_gt)
    
    # Log final results
    logging.info(f'Sample-{args.sampID}: data_iter {data_iter_step}, initial acc {acc_samp_init:.4f}, TTT acc {acc_samp_TTT:.4f}')
    logging.info(f"Sample-{args.sampID}: Hit {hit_cnt_TTT}/{num_samples} after TTT, Hit {hit_cnt_init}/{num_samples} at init")
    logging.info(f'TTT results: batch size {batch_wavefields}, steps per example {args.steps_per_example}, lr {args.lr}')
    
    # Save results
    _save_training_results(args, all_results, all_losses, acc_samp_init, acc_samp_TTT, batch_wavefields)
    
    return acc_samp_TTT


def _save_training_results(args: argparse.Namespace,
                         all_results: torch.Tensor,
                         all_losses: torch.Tensor,
                         acc_samp_init: float,
                         acc_samp_TTT: float,
                         batch_size: int) -> None:
    """
    Save training results and metrics to files.
    
    Args:
        args: Parsed command line arguments
        all_results: Accuracy results tensor
        all_losses: Loss results tensor  
        acc_samp_init: Initial accuracy
        acc_samp_TTT: Final TTT accuracy
        batch_size: Batch size used for training
    """
    # Save accuracy and loss arrays
    accuracy_path = os.path.join(args.output_dir, f'Sample_{args.sampID}_accuracy_itr_{args.steps_per_example}.npy')
    loss_path = os.path.join(args.output_dir, f'Sample_{args.sampID}_loss_itr_{args.steps_per_example}.npy')
    
    with open(accuracy_path, 'wb') as f:
        np.save(f, np.array(all_results))
    with open(loss_path, 'wb') as f:
        np.save(f, np.array(all_losses))
    
    # Append summary to tracking file
    summary_path = os.path.join(args.output_dir, 'accuracy_TTT_saves.txt')
    with open(summary_path, 'a') as f:
        f.write(f'Sample-{args.sampID}: batch-{batch_size}, iter-{args.steps_per_example}, '
                f'lr-{args.lr}, init acc {acc_samp_init:.4f}, TTT acc {acc_samp_TTT:.4f}\n')
    
    logging.info(f"Results saved to {accuracy_path} and {loss_path}")


def save_bn_statistics_history(args: argparse.Namespace, bn_stats_history: List[Dict]) -> None:
    """Save BN statistics history for analysis"""
    import pickle
    with open(os.path.join(args.output_dir, f'Samp_{args.sampID}_bn_stats_history.pkl'), 'wb') as f:
        pickle.dump(bn_stats_history, f)

def plot_bn_statistics_evolution(bn_stats_history, sampID, save_path=None):
    """Plot evolution of BN statistics over time"""
    if not bn_stats_history:
        return
        
    plt.figure(figsize=(15, 10))
    
    # Extract first BN layer for visualization
    first_bn_name = list(bn_stats_history[0]['stats'].keys())[0]
    
    data_iters = [entry['data_iter'] for entry in bn_stats_history]
    
    # Plot running mean evolution
    plt.subplot(2, 2, 1)
    running_means = [entry['stats'][first_bn_name]['running_mean'].cpu().numpy() for entry in bn_stats_history]
    for i in range(min(5, len(running_means[0]))):  # Plot first 5 channels
        means_channel = [mean[i] for mean in running_means]
        plt.plot(data_iters, means_channel, label=f'Channel {i}')
    plt.title(f'BN Running Mean Evolution - {first_bn_name}')
    plt.xlabel('Data Iteration')
    plt.ylabel('Running Mean')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot running variance evolution
    plt.subplot(2, 2, 2)
    running_vars = [entry['stats'][first_bn_name]['running_var'].cpu().numpy() for entry in bn_stats_history]
    for i in range(min(5, len(running_vars[0]))):  # Plot first 5 channels
        vars_channel = [var[i] for var in running_vars]
        plt.plot(data_iters, vars_channel, label=f'Channel {i}')
    plt.title(f'BN Running Variance Evolution - {first_bn_name}')
    plt.xlabel('Data Iteration')
    plt.ylabel('Running Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot momentum evolution
    plt.subplot(2, 2, 3)
    momentums = [entry['stats'][first_bn_name]['momentum'] for entry in bn_stats_history]
    plt.plot(data_iters, momentums, 'r-', linewidth=2)
    plt.title('BN Momentum Evolution')
    plt.xlabel('Data Iteration')
    plt.ylabel('Momentum')
    plt.grid(True, alpha=0.3)
    
    # Plot mean of running means across all channels
    plt.subplot(2, 2, 4)
    mean_of_means = [np.mean(mean) for mean in running_means]
    mean_of_vars = [np.mean(var) for var in running_vars]
    plt.plot(data_iters, mean_of_means, 'b-', label='Mean of Running Means', linewidth=2)
    plt.plot(data_iters, mean_of_vars, 'g-', label='Mean of Running Vars', linewidth=2)
    plt.title('Overall BN Statistics Trend')
    plt.xlabel('Data Iteration')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Batch Normalization Statistics Evolution - Sample {sampID}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"BN statistics evolution plot saved to: {save_path}")
    
    plt.show()
    plt.close()
    


def save_accuracy_results(args):
    all_all_results = [list() for i in range(args.steps_per_example)]
    for file_number, f_name in enumerate(glob.glob(os.path.join(args.output_dir, 'results_*.npy'))):
        all_data = np.load(f_name)
        for step in range(args.steps_per_example):
            all_all_results[step] += all_data[step].tolist()
    with open(os.path.join(args.output_dir, 'model-final.pth'), 'w') as f:
        f.write(f'Done!\n')
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:
        f.write(f'{str(args)}\n')
        for i in range(args.steps_per_example):
            # assert len(all_all_results[i]) == 50000, len(all_all_results[i])
            f.write(f'{i}\t{np.mean(all_all_results[i])}\n')


def save_results(args):
    # all_all_results = [list() for i in range(args.steps_per_example)]
    # for file_number, f_name in enumerate(glob.glob(os.path.join(args.output_dir, 'results_*.npy'))):
    #     all_data = np.load(f_name)
    #     for step in range(args.steps_per_example):
    #         all_all_results[step] += all_data[step].tolist()
 
    fName_acc = os.path.join(args.output_dir, f'Sample_{args.sampID}_accuracy_itr_{args.steps_per_example}.npy')

    fName_loss = os.path.join(args.output_dir, f'Sample_{args.sampID}_loss_itr_{args.steps_per_example}.npy')    
    all_acc = np.load(fName_acc)
    all_loss = np.load(fName_loss)    
    
    # with open(os.path.join(args.output_dir, 'model-final.pth'), 'w') as f:
        # f.write(f'Done!\n')
    with open(os.path.join(args.output_dir, 'accuracy.txt'), 'a') as f:


        # f.write(f'{str(args)}\n')
        # for i in range(args.steps_per_example):
            # assert len(all_all_results[i]) == 50000, len(all_all_results[i])
            # f.write(f'{i}\t{np.mean(all_acc).item()}\n')                        

        f.write(f'{all_acc.mean(axis=0)}')
