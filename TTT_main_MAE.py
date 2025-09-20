# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Test-Time Training with Masked Autoencoders for Ultrasound Image Analysis
# 
# This implementation provides test-time training (TTT) capabilities using
# Masked Autoencoders (MAE) with Vision Transformers for ultrasound image
# classification tasks. The system adapts pre-trained models to test samples
# through online LayerNorm updates and self-supervised reconstruction.
#
# References:
# - DeiT: https://github.com/facebookresearch/deit
# - BEiT: https://github.com/microsoft/unilm/tree/master/beit
# - MAE: He, K., et al. "Masked autoencoders are scalable vision learners." CVPR 2022.
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import logging

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

import timm
from torchvision import datasets
import glob

# Local imports
import util.misc as misc
import models_mae_shared
from engine_TTT_wavefield_LN_MAE_vis import train_on_test_with_online_ln_update_MAE, get_prameters_from_args
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args_parser() -> argparse.ArgumentParser:
    """
    Configure command-line arguments for Test-Time Training with MAE.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all TTT-MAE parameters
    """
    parser = argparse.ArgumentParser(
        'MAE Test-Time Training for Ultrasound Image Analysis', 
        add_help=True,
        description='Train and adapt Masked Autoencoders at test time for improved ultrasound image classification'
    )
    
    # ========================================
    # Model Architecture Parameters
    # ========================================
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model', 
        default='mae_vit_base_patch16', 
        type=str, 
        metavar='MODEL',
        help='Model architecture name (default: mae_vit_base_patch16)'
    )
    model_group.add_argument(
        '--input_size', 
        default=224, 
        type=int,
        help='Input image size in pixels (default: 224)'
    )
    model_group.add_argument(
        '--classifier_depth', 
        type=int, 
        metavar='N', 
        default=0,
        help='Number of blocks in the classifier head (default: 0)'
    )
    model_group.add_argument(
        '--head_type', 
        default='vit_head',
        choices=['linear', 'vit_head'],
        help='Type of classification head (default: vit_head)'
    )
    
    # ========================================
    # Test-Time Training Parameters
    # ========================================
    ttt_group = parser.add_argument_group('Test-Time Training Configuration')
    ttt_group.add_argument(
        '--mask_ratio', 
        default=0.75, 
        type=float,
        help='Masking ratio for MAE reconstruction (default: 0.75)'
    )
    ttt_group.add_argument(
        '--finetune_mode', 
        default='encoder', 
        type=str,
        choices=['all', 'encoder', 'encoder_no_cls_no_msk'],
        help='Fine-tuning strategy (default: encoder)'
    )
    ttt_group.add_argument(
        '--steps_per_example', 
        default=64, 
        type=int,
        help='Number of adaptation steps per test example (default: 64)'
    )
    
    # ========================================
    # Training and Optimization Parameters
    # ========================================
    optim_group = parser.add_argument_group('Optimization Configuration')
    optim_group.add_argument(
        '--batch_size', 
        default=16, 
        type=int,
        help='Batch size for training (default: 16)'
    )
    optim_group.add_argument(
        '--blr', 
        type=float, 
        default=1e-3, 
        metavar='LR',
        help='Base learning rate (default: 1e-3)'
    )
    optim_group.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.05,
        help='Weight decay regularization (default: 0.05)'
    )
    optim_group.add_argument(
        '--optimizer_type', 
        default='sgd',
        choices=['adam', 'adam_w', 'sgd'],
        help='Optimizer type (default: sgd)'
    )
    optim_group.add_argument(
        '--optimizer_momentum', 
        default=0.9, 
        type=float,
        help='Momentum for SGD optimizer (default: 0.9)'
    )
    optim_group.add_argument(
        '--accum_iter', 
        default=1, 
        type=int,
        help='Gradient accumulation iterations (default: 1)'
    )
    
    # ========================================
    # Data and Preprocessing Parameters  
    # ========================================
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        '--data_path', 
        default='', 
        type=str,
        help='Path to dataset directory'
    )
    data_group.add_argument(
        '--dataset_name', 
        default='imagenet_c', 
        type=str,
        help='Dataset name for identification'
    )
    data_group.add_argument(
        '--single_crop', 
        action='store_true',
        help='Use single crop instead of random crop for training'
    )
    data_group.add_argument(
        '--no_single_crop', 
        action='store_false', 
        dest='single_crop'
    )
    parser.set_defaults(single_crop=False)
    
    # ========================================
    # Model Loading and Checkpoints
    # ========================================
    checkpoint_group = parser.add_argument_group('Model Checkpoints')
    checkpoint_group.add_argument(
        '--resume_model', 
        default='vit-B-ft-US.pt',
        help='Path to base model checkpoint (default: vit-B-ft-US.pt)'
    )
    checkpoint_group.add_argument(
        '--resume_finetune', 
        default='vit-B-ft-US.pt',
        help='Path to fine-tuned model checkpoint (default: vit-B-ft-US.pt)'
    )
    checkpoint_group.add_argument(
        '--stored_latents', 
        default='',
        help='Path to pre-computed latent features (optional)'
    )
    checkpoint_group.add_argument(
        '--load_loss_scalar', 
        action='store_true',
        help='Load loss scaler from checkpoint'
    )
    parser.set_defaults(load_loss_scalar=False)
    
    # ========================================
    # Output and Logging Parameters
    # ========================================
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output_dir', 
        default='./output_dir',
        help='Directory for saving results (default: ./output_dir)'
    )
    output_group.add_argument(
        '--log_dir', 
        default='./output_dir',
        help='Directory for tensorboard logs (default: ./output_dir)'
    )
    output_group.add_argument(
        '--print_freq', 
        default=50, 
        type=int,
        help='Frequency of progress printing (default: 50)'
    )
    output_group.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    output_group.add_argument(
        '--debug_mode', 
        action='store_true',
        help='Enable debug mode'
    )
    parser.set_defaults(verbose=False)
    
    # ========================================
    # System and Hardware Parameters
    # ========================================
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument(
        '--device', 
        default='mps',
        help='Device for computation (default: mps)'
    )
    system_group.add_argument(
        '--seed', 
        default=0, 
        type=int,
        help='Random seed for reproducibility (default: 0)'
    )
    system_group.add_argument(
        '--num_workers', 
        default=0, 
        type=int,
        help='Number of data loading workers (default: 0)'
    )
    system_group.add_argument(
        '--pin_mem', 
        action='store_true',
        help='Pin CPU memory in DataLoader for efficient GPU transfer'
    )
    system_group.add_argument(
        '--no_pin_mem', 
        action='store_false', 
        dest='pin_mem'
    )
    parser.set_defaults(pin_mem=False)
    
    # ========================================
    # Advanced Training Parameters
    # ========================================
    advanced_group = parser.add_argument_group('Advanced Configuration')
    advanced_group.add_argument(
        '--norm_pix_loss', 
        action='store_true',
        help='Use per-patch normalized pixels for loss computation'
    )
    parser.set_defaults(norm_pix_loss=False)
    
    # ========================================
    # Distributed Training Parameters
    # ========================================
    dist_group = parser.add_argument_group('Distributed Training')
    dist_group.add_argument(
        '--world_size', 
        default=1, 
        type=int,
        help='Number of distributed processes (default: 1)'
    )
    dist_group.add_argument(
        '--local_rank', 
        default=-1, 
        type=int,
        help='Local rank for distributed training'
    )
    dist_group.add_argument(
        '--dist_on_itp', 
        action='store_true',
        help='Enable distributed training on ITP'
    )
    dist_group.add_argument(
        '--dist_url', 
        default='env://',
        help='URL for distributed training setup (default: env://)'
    )

    return parser


class CustomTensorDataset(Dataset):
    """
    Custom PyTorch Dataset for handling ultrasound image tensors.
    
    This dataset class provides proper preprocessing and transformation
    for ultrasound image data, handling various tensor formats and
    applying necessary augmentations.
    
    Args:
        tensors (torch.Tensor): Input image tensors
        targets (torch.Tensor): Corresponding target labels  
        transform (Optional[transforms.Compose]): Image transformations to apply
    """
    
    def __init__(
        self, 
        tensors: torch.Tensor, 
        targets: torch.Tensor, 
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize the dataset with tensors, targets, and optional transforms."""
        self.tensors = tensors
        self.targets = targets
        self.transform = transform
        
        # Validate input shapes
        assert len(tensors) == len(targets), (
            f"Tensor and target lengths must match: {len(tensors)} vs {len(targets)}"
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed image tensor and target label
        """
        x = self.tensors[index]
        y = self.targets[index]
        
        # Convert tensor to numpy array for PIL transformations
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Handle different tensor formats
        x = self._preprocess_image(x)
        
        # Apply transformations if specified
        if self.transform:
            x_pil = transforms.ToPILImage()(x)
            x = self.transform(x_pil)
            
        return x, y

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.tensors)
    
    def _preprocess_image(self, x: np.ndarray) -> np.ndarray:
        """
        Preprocess image array to ensure proper format for PIL operations.
        
        Args:
            x (np.ndarray): Input image array
            
        Returns:
            np.ndarray: Preprocessed image array in (H, W, C) format
        """
        # Add channel dimension for grayscale images (H, W) -> (H, W, 1)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        
        # Convert from (C, H, W) to (H, W, C) format if needed
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.transpose(1, 2, 0)
                            
        return x

def load_combined_model(args: argparse.Namespace, num_classes: int = 1000) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[NativeScaler]]:
    """
    Load and initialize the combined MAE model with proper checkpoints.
    
    This function handles loading both the base model and fine-tuned checkpoints,
    properly configuring the model architecture based on the specified parameters.
    
    Args:
        args (argparse.Namespace): Configuration arguments
        num_classes (int): Number of output classes for classification
        
    Returns:
        Tuple containing:
            - torch.nn.Module: Loaded and configured model
            - Optional[torch.optim.Optimizer]: Optimizer (None in this implementation)
            - Optional[NativeScaler]: Loss scaler if requested, None otherwise
            
    Raises:
        AssertionError: If unsupported model architecture is specified
        FileNotFoundError: If checkpoint files cannot be found
    """
    # Configure model architecture parameters based on model type
    model_configs = {
        'mae_vit_small_patch16': {
            'classifier_depth': 8,
            'classifier_embed_dim': 512,
            'classifier_num_heads': 16
        },
        'mae_vit_base_patch16': {
            'classifier_depth': 12,
            'classifier_embed_dim': 768,
            'classifier_num_heads': 12
        },
        'mae_vit_large_patch16': {
            'classifier_depth': 12,
            'classifier_embed_dim': 768,
            'classifier_num_heads': 12
        },
        'mae_vit_huge_patch14': {
            'classifier_depth': 12,
            'classifier_embed_dim': 768,
            'classifier_num_heads': 12
        }
    }
    
    # Validate model architecture
    if args.model not in model_configs:
        supported_models = list(model_configs.keys())
        raise ValueError(f"Unsupported model '{args.model}'. Supported models: {supported_models}")
    
    config = model_configs[args.model]
    
    # Initialize model with appropriate configuration
    try:
        model = models_mae_shared.__dict__[args.model](
            num_classes=num_classes, 
            head_type=args.head_type, 
            norm_pix_loss=args.norm_pix_loss,
            classifier_depth=config['classifier_depth'],
            classifier_embed_dim=config['classifier_embed_dim'],
            classifier_num_heads=config['classifier_num_heads'],
            rec_opt=False,
            rotation_prediction=False
        )
        logger.info(f"Successfully initialized {args.model} with {num_classes} classes")
    except Exception as e:
        logger.error(f"Failed to initialize model {args.model}: {e}")
        raise
    
    # Load model checkpoints
    try:
        model_checkpoint = torch.load(args.resume_model, map_location='cpu', weights_only=False)
        head_checkpoint = torch.load(args.resume_finetune, map_location='cpu', weights_only=False)
        logger.info(f"Loaded checkpoints: {args.resume_model}, {args.resume_finetune}")
    except FileNotFoundError as e:
        logger.error(f"Checkpoint file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoints: {e}")
        raise
    
    # Extract state dictionary from checkpoint
    head_state_dict = _extract_state_dict(head_checkpoint)
    
    # Handle different head types for backward compatibility
    if args.head_type == 'linear':
        _load_linear_head_weights(model_checkpoint, head_checkpoint)
    elif args.classifier_depth != 0:
        _load_classifier_weights(model_checkpoint, head_checkpoint)
        
    # Load state dictionary into model
    try:
        missing_keys, unexpected_keys = model.load_state_dict(head_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys during model loading: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys during model loading: {unexpected_keys[:5]}...")
            
        logger.info("Model state dictionary loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model state dictionary: {e}")
        raise
    
    # Initialize optimizer (None in this implementation)
    optimizer = None
    
    # Initialize loss scaler if requested
    loss_scaler = None
    if args.load_loss_scalar:
        try:
            loss_scaler = NativeScaler()
            if 'scaler' in model_checkpoint:
                loss_scaler.load_state_dict(model_checkpoint['scaler'])
                logger.info("Loss scaler loaded from checkpoint")
            else:
                logger.warning("Loss scaler requested but not found in checkpoint")
        except Exception as e:
            logger.error(f"Error loading loss scaler: {e}")
            loss_scaler = None
    
    return model, optimizer, loss_scaler


def _extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Extract state dictionary from checkpoint with flexible format handling.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Dict[str, torch.Tensor]: Extracted state dictionary
    """
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    elif hasattr(checkpoint, 'state_dict'):
        return checkpoint.state_dict()
    else:
        return checkpoint


def _load_linear_head_weights(model_checkpoint: Dict[str, Any], head_checkpoint: Dict[str, Any]) -> None:
    """
    Load linear head weights into model checkpoint.
    
    Args:
        model_checkpoint: Main model checkpoint dictionary
        head_checkpoint: Head-specific checkpoint dictionary
    """
    try:
        model_checkpoint['model']['bn.running_mean'] = head_checkpoint['model']['head.0.running_mean']
        model_checkpoint['model']['bn.running_var'] = head_checkpoint['model']['head.0.running_var']
        model_checkpoint['model']['head.weight'] = head_checkpoint['model']['head.1.weight']
        model_checkpoint['model']['head.bias'] = head_checkpoint['model']['head.1.bias']
        logger.debug("Linear head weights loaded successfully")
    except KeyError as e:
        logger.error(f"Missing key when loading linear head weights: {e}")
        raise


def _load_classifier_weights(model_checkpoint: Dict[str, Any], head_checkpoint: Dict[str, Any]) -> None:
    """
    Load classifier weights into model checkpoint.
    
    Args:
        model_checkpoint: Main model checkpoint dictionary  
        head_checkpoint: Head-specific checkpoint dictionary
    """
    try:
        for key in head_checkpoint['model']:
            if key.startswith('classifier'):
                model_checkpoint['model'][key] = head_checkpoint['model'][key]
        logger.debug("Classifier weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading classifier weights: {e}")
        raise

def setup_training_environment(args: argparse.Namespace) -> None:
    """
    Set up the training environment with proper seed, device, and logging.
    
    Args:
        args (argparse.Namespace): Configuration arguments
    """
    # Set up directories
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    
    # Set device and random seeds for reproducibility
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info(f"Using device: {device}, seed: {seed}")
    
    # Enable CuDNN benchmarking for performance
    cudnn.benchmark = True
    
    # Log job directory and configuration
    logger.info(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info("Configuration:")
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")


def setup_data_transforms(args: argparse.Namespace) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Set up data transformation pipelines for training and validation.
    
    Args:
        args (argparse.Namespace): Configuration arguments
        
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Training and validation transforms
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    # Validation transforms (always center crop)
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),  # Bicubic interpolation
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        normalize
    ])
    
    # Training transforms (data augmentation vs single crop)
    if not args.single_crop:
        # Standard data augmentation pipeline
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(
                args.input_size, 
                scale=(0.2, 1.0), 
                interpolation=3  # Bicubic
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Single crop (same as validation)
        transform_train = transform_val
        
    logger.info(f"Data transforms configured: single_crop={args.single_crop}")
    return transform_train, transform_val


def load_ultrasound_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load ultrasound data from the data reading script.
    
    Returns:
        Tuple containing:
            - USIpipeNG: Pipe NG ultrasound images
            - USIpipeNGlabel: Pipe NG labels
            - USIpipeNGidx: Pipe NG sample indices
            - USIplateNG: Plate NG ultrasound images  
            - USIplateNGlabel: Plate NG labels
    """
    # Load data using the existing script
    logger.info("Loading ultrasound data...")
    
    # Check if data is already loaded in global scope
    if 'USIplateOK' not in globals():
        logger.info("Executing USdataRead.py to load data...")
        exec(open('USdataRead.py').read(), globals())
    
    # Access global variables
    global USIplateNG, USIplateNGlabel, USIpipeNG, USIpipeNGlabel, USIpipeNGidx
    
    # Ensure proper tensor formats
    USIplateNGlabel = USIplateNGlabel.flatten().long()
    USIpipeNGlabel = USIpipeNGlabel.flatten().long()
    USIpipeNGidx = USIpipeNGidx.flatten().long()
    
    logger.info(f"Loaded ultrasound data:")
    logger.info(f"  - Pipe NG images: {USIpipeNG.shape}")
    logger.info(f"  - Pipe NG labels: {USIpipeNGlabel.shape}")
    logger.info(f"  - Pipe NG indices: {USIpipeNGidx.shape}")
    logger.info(f"  - Plate NG images: {USIplateNG.shape}")
    logger.info(f"  - Plate NG labels: {USIplateNGlabel.shape}")
    
    return USIpipeNG, USIpipeNGlabel, USIpipeNGidx, USIplateNG, USIplateNGlabel


def process_sample(
    sample_id: int,
    args: argparse.Namespace,
    pipe_images: torch.Tensor,
    pipe_labels: torch.Tensor,
    pipe_indices: torch.Tensor,
    transform_val: transforms.Compose,
    num_classes: int
) -> float:
    """
    Process a single ultrasound sample with test-time training.
    
    Args:
        sample_id (int): Unique identifier for the sample
        args (argparse.Namespace): Configuration arguments
        pipe_images (torch.Tensor): Pipe ultrasound images
        pipe_labels (torch.Tensor): Pipe labels
        pipe_indices (torch.Tensor): Sample indices
        transform_val (transforms.Compose): Validation transforms
        num_classes (int): Number of classification classes
        
    Returns:
        float: Test-time training accuracy for this sample
    """
    logger.info(f"Processing sample {sample_id}...")
    
    # Filter data for current sample
    sample_mask = pipe_indices == sample_id
    sample_images = pipe_images[sample_mask]
    sample_labels = pipe_labels[sample_mask]
    
    if len(sample_images) == 0:
        logger.warning(f"No data found for sample {sample_id}")
        return 0.0
    
    logger.info(f"Sample {sample_id}: {len(sample_images)} images")
    
    # Create dataset and dataloader
    sample_dataset = CustomTensorDataset(
        sample_images, 
        sample_labels, 
        transform=transform_val
    )
    
    # Store sample ID in args for logging
    args.sampID = int(sample_id)
    
    # Load and initialize model
    base_model, base_optimizer, base_scalar = load_combined_model(args, num_classes)
    
    # Calculate effective batch size and learning rate
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    args.lr = args.blr  # Use base learning rate directly
    
    logger.info(f"Training configuration for sample {sample_id}:")
    logger.info(f"  - Effective batch size: {eff_batch_size}")
    logger.info(f"  - Learning rate: {args.lr}")
    logger.info(f"  - Accumulation iterations: {args.accum_iter}")
    
    # Record start time
    start_time = time.time()
    
    # Perform test-time training
    try:
        ttt_accuracy = train_on_test_with_online_ln_update_MAE(
            base_model, 
            base_optimizer, 
            base_scalar, 
            sample_dataset,  # training dataset
            sample_dataset,  # validation dataset  
            torch.device(args.device),
            log_writer=None, 
            args=args, 
            num_classes=num_classes
        )
        
        # Calculate processing time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        
        logger.info(f"Sample {sample_id} completed:")
        logger.info(f"  - TTT accuracy: {ttt_accuracy:.4f}")
        logger.info(f"  - Processing time: {total_time_str}")
        
        # Log results to file
        _log_sample_results(args, sample_id, ttt_accuracy, total_time)
        
        return ttt_accuracy
        
    except Exception as e:
        logger.error(f"Error processing sample {sample_id}: {e}")
        return 0.0


def _log_sample_results(args: argparse.Namespace, sample_id: int, accuracy: float, processing_time: float) -> None:
    """
    Log individual sample results to file.
    
    Args:
        args (argparse.Namespace): Configuration arguments
        sample_id (int): Sample identifier
        accuracy (float): Achieved accuracy
        processing_time (float): Processing time in seconds
    """
    log_filename = f'Scan_b_{args.batch_size}_itr_{args.steps_per_example}_lr_{int(args.lr*10000)}d10000.txt'
    log_path = os.path.join(args.output_dir, log_filename)
    
    with open(log_path, 'a') as f:
        f.write(
            f'Sample-{sample_id}: batch-{args.batch_size}, '
            f'iter-{args.steps_per_example}, lr-{args.lr}, '
            f'TTT acc {accuracy:.4f}, proc time {processing_time:.2f}\n'
        )


def log_final_results(args: argparse.Namespace, all_accuracies: List[float], all_times: List[float]) -> None:
    """
    Log final aggregated results.
    
    Args:
        args (argparse.Namespace): Configuration arguments
        all_accuracies (List[float]): List of all sample accuracies
        all_times (List[float]): List of all processing times
    """
    if not all_accuracies:
        logger.warning("No results to log")
        return
        
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_time = np.mean(all_times)
    std_time = np.std(all_times)
    
    logger.info("Final Results Summary:")
    logger.info(f"  - Mean TTT Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"  - Mean Processing Time: {mean_time:.2f} ± {std_time:.2f} seconds")
    logger.info(f"  - Total Samples Processed: {len(all_accuracies)}")
    
    # Log to file
    log_filename = f'Scan_b_{args.batch_size}_itr_{args.steps_per_example}_lr_{int(args.lr*10000)}d10000.txt'
    log_path = os.path.join(args.output_dir, log_filename)
    
    with open(log_path, 'a') as f:
        f.write(
            f'FINAL SUMMARY: batch-{args.batch_size}, '
            f'iter-{args.steps_per_example}, lr-{args.lr}, '
            f'TTT acc. avg {mean_accuracy:.4f}, std {std_accuracy:.4f}, '
            f'time avg {mean_time:.2f}, std {std_time:.2f}\n'
        )


def main(args: argparse.Namespace) -> None:
    """
    Main function for Test-Time Training with Masked Autoencoders.
    
    This function orchestrates the entire TTT-MAE pipeline including:
    - Environment setup and data loading
    - Model initialization and configuration
    - Per-sample test-time training
    - Results aggregation and logging
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    logger.info("Starting Test-Time Training with Masked Autoencoders")
    
    # Set up training environment
    setup_training_environment(args)
    
    # Set up data transforms
    transform_train, transform_val = setup_data_transforms(args)
    
    # Set LayerNorm adaptation parameters
    args.ln_momentum = getattr(args, 'ln_momentum', 0.05)  # Lower for slower adaptation
    args.ln_update_mode = getattr(args, 'ln_update_mode', 'adaptive')  # Adaptive momentum
    args.preserve_ln_across_samples = getattr(args, 'preserve_ln_across_samples', True)
    
    # Configuration for ultrasound classification
    num_classes = 2
    logger.info(f"Configured for {num_classes}-class ultrasound classification")
    
    # Load ultrasound data
    USIpipeNG, USIpipeNGlabel, USIpipeNGidx, USIplateNG, USIplateNGlabel = load_ultrasound_data()
    
    # Process each unique sample
    unique_samples = USIpipeNGidx.unique()
    logger.info(f"Found {len(unique_samples)} unique samples to process")
    
    all_accuracies = []
    all_times = []
    
    for sample_id in unique_samples:
        try:
            ttt_accuracy = process_sample(
                sample_id=int(sample_id),
                args=args,
                pipe_images=USIpipeNG,
                pipe_labels=USIpipeNGlabel,
                pipe_indices=USIpipeNGidx,
                transform_val=transform_val,
                num_classes=num_classes
            )
            
            all_accuracies.append(ttt_accuracy)
            
        except Exception as e:
            logger.error(f"Failed to process sample {sample_id}: {e}")
            continue
    
    # Log final aggregated results
    log_final_results(args, all_accuracies, all_times)
    
    logger.info("Test-Time Training completed successfully")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run main function
    main(args)
