"""
train.py - Training Pipeline for Dun Huang Mural Restoration

This module implements the complete training pipeline for the mural restoration model,
including data loading, loss functions, optimization, and evaluation. 

Key components:
- MuralDataset: Custom dataset for damaged/restored mural pairs
- MuralLoss: Combined loss function with perceptual and structural components
- MuralTrainer: Training loop with validation and checkpoint management

Training flow:
1. Load and preprocess Dun Huang Mural dataset
2. Initialize model, losses, and optimizer
3. Execute training loop with periodic validation
4. Save best models based on validation metrics

Usage:
    python train.py --data_path /path/to/murals 
                    --batch_size 8 
                    --epochs 200 
                    --lr 0.0001
"""

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image

# Import the updated model components
from encoder import SwinEncoderExtractor
from decoder import DecoderModule
from model import MuralRestorationModel


class MuralDataset(Dataset):
    """
    Dataset for Dun Huang Mural restoration.
    
    This dataset loads pairs of damaged and ground truth mural images
    for training the restoration model. It handles data augmentation
    and preprocessing specific to the mural restoration task.
    
    Attributes:
        data_path (str): Path to the dataset directory
        mode (str): 'train', 'val', or 'test'
        transform (transforms.Compose): Data augmentation and preprocessing
        damaged_paths (List[str]): Paths to damaged mural images
        gt_paths (List[str]): Paths to ground truth mural images
    """
    
    def __init__(
        self,
        data_path: str,
        mode: str = 'train',
        image_size: int = 256,
        use_augmentation: bool = True
    ):
        """
        Initialize the mural dataset.
        
        Args:
            data_path (str): Path to the dataset directory
            mode (str): 'train', 'val', or 'test'
            image_size (int): Size to resize images to
            use_augmentation (bool): Whether to use data augmentation (for training only)
        """
        super(MuralDataset, self).__init__()
        
        self.data_path = data_path
        self.mode = mode
        self.image_size = image_size
        
        # Set up data paths
        mode_folder = os.path.join(data_path, mode)
        damaged_folder = os.path.join(mode_folder, 'damaged')
        gt_folder = os.path.join(mode_folder, 'ground_truth')
        
        # Get image paths
        self.damaged_paths = sorted([
            os.path.join(damaged_folder, f) for f in os.listdir(damaged_folder)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.gt_paths = sorted([
            os.path.join(gt_folder, f) for f in os.listdir(gt_folder)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        # Verify matching pairs
        assert len(self.damaged_paths) == len(self.gt_paths), \
            f"Mismatched number of images: {len(self.damaged_paths)} damaged, {len(self.gt_paths)} ground truth"
        
        # Set up transformations
        self.transform = self._get_transforms(mode, image_size, use_augmentation)
        
        print(f"Loaded {len(self.damaged_paths)} {mode} image pairs")
    
    def _get_transforms(self, mode: str, image_size: int, use_augmentation: bool) -> transforms.Compose:
        """
        Get image transformations based on dataset mode.
        
        Args:
            mode (str): 'train', 'val', or 'test'
            image_size (int): Size to resize images to
            use_augmentation (bool): Whether to use data augmentation
            
        Returns:
            transforms.Compose: Transformation pipeline
        """
        # Basic transformations for all modes
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        
        # Add augmentations for training
        if mode == 'train' and use_augmentation:
            augmentation = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)
            ]
            transform_list = augmentation + transform_list
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.damaged_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a damaged/ground truth image pair.
        
        Args:
            idx (int): Index of the image pair
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'damaged' and 'gt' tensors
        """
        # Load images
        damaged_image = transforms.Image.open(self.damaged_paths[idx]).convert('RGB')
        gt_image = transforms.Image.open(self.gt_paths[idx]).convert('RGB')
        
        # Apply transformations
        seed = np.random.randint(2147483647)  # Use same seed for both images
        
        torch.manual_seed(seed)
        damaged_tensor = self.transform(damaged_image)
        
        torch.manual_seed(seed)
        gt_tensor = self.transform(gt_image)
        
        return {
            'damaged': damaged_tensor,
            'gt': gt_tensor,
            'path': os.path.basename(self.damaged_paths[idx])
        }


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    
    This loss computes the L1 distance between feature maps extracted
    from pretrained VGG16 for both predicted and ground truth images.
    It helps to preserve semantic and structural information.
    
    Attributes:
        vgg (nn.Module): Truncated VGG16 model for feature extraction
        layer_weights (Dict[str, float]): Weights for different VGG layers
        criterion (nn.Module): Distance function (L1Loss)
    """
    
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the perceptual loss.
        
        Args:
            layer_weights (Optional[Dict[str, float]]): Weights for different VGG layers
        """
        super(PerceptualLoss, self).__init__()
        
        # Use default weights if not provided
        if layer_weights is None:
            self.layer_weights = {
                'relu1_2': 0.1,
                'relu2_2': 0.2,
                'relu3_3': 0.4,
                'relu4_3': 0.3
            }
        else:
            self.layer_weights = layer_weights
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Define layer mapping
        self.layer_map = {
            '3': 'relu1_2',   # After 2nd conv of 1st block
            '8': 'relu2_2',   # After 2nd conv of 2nd block
            '15': 'relu3_3',  # After 3rd conv of 3rd block
            '22': 'relu4_3'   # After 3rd conv of 4th block
        }
        
        # Register forward hooks to extract features
        self.outputs = {}
        for name, module in vgg._modules.items():
            if name in self.layer_map:
                module.register_forward_hook(self._get_features_hook(name))
        
        # Move to GPU if available
        self.vgg = vgg.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.criterion = nn.L1Loss()
    
    def _get_features_hook(self, name: str):
        """
        Create a hook function to capture feature maps.
        
        Args:
            name (str): Name of the layer
            
        Returns:
            callable: Hook function
        """
        def hook(module, input, output):
            self.outputs[name] = output
        return hook
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the perceptual loss between predicted and target images.
        
        Args:
            predicted (torch.Tensor): Predicted image [B, 3, H, W]
            target (torch.Tensor): Ground truth image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Perceptual loss value
        """
        # Normalize to VGG16 input range
        predicted = (predicted + 1) / 2  # Convert from [-1, 1] to [0, 1]
        target = (target + 1) / 2
        
        # Extract features
        self.outputs.clear()
        _ = self.vgg(predicted)
        predicted_features = {self.layer_map[k]: v for k, v in self.outputs.items()}
        
        self.outputs.clear()
        _ = self.vgg(target)
        target_features = {self.layer_map[k]: v for k, v in self.outputs.items()}
        
        # Compute weighted loss
        loss = 0.0
        for layer_name, weight in self.layer_weights.items():
            loss += weight * self.criterion(
                predicted_features[layer_name], 
                target_features[layer_name]
            )
        
        return loss


class MuralLoss(nn.Module):
    """
    Combined loss function for mural restoration.
    
    This loss combines multiple components:
    1. L1 loss for pixel accuracy
    2. SSIM loss for structural similarity
    3. Perceptual loss for semantic content
    4. Edge-aware loss for fine details
    
    Attributes:
        l1_weight (float): Weight for L1 loss
        ssim_weight (float): Weight for SSIM loss
        perceptual_weight (float): Weight for perceptual loss
        edge_weight (float): Weight for edge-aware loss
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        edge_weight: float = 0.2
    ):
        """
        Initialize the combined loss.
        
        Args:
            l1_weight (float): Weight for L1 loss
            ssim_weight (float): Weight for SSIM loss
            perceptual_weight (float): Weight for perceptual loss
            edge_weight (float): Weight for edge-aware loss
        """
        super(MuralLoss, self).__init__()
        
        # Store weights
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.edge_weight = edge_weight
        
        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
    
    def _ssim_loss(self, predicted: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """
        Compute the structural similarity loss.
        
        Args:
            predicted (torch.Tensor): Predicted image [B, 3, H, W]
            target (torch.Tensor): Ground truth image [B, 3, H, W]
            window_size (int): Size of the SSIM window
            
        Returns:
            torch.Tensor: 1 - SSIM (as a loss function)
        """
        # Convert from [-1, 1] to [0, 1]
        predicted = (predicted + 1) / 2
        target = (target + 1) / 2
        
        # Parameters for SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / (2 * sigma**2))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        # 2D Gaussian window
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(3, 1, window_size, window_size).contiguous()
        window = window.to(predicted.device)
        
        # Mean, variance, covariance
        mu1 = F.conv2d(predicted, window, padding=window_size//2, groups=3)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=3)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(predicted * predicted, window, padding=window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(predicted * target, window, padding=window_size//2, groups=3) - mu1_mu2
        
        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return 1 - SSIM as a loss
        return 1 - ssim_map.mean()
    
    def _edge_loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the edge-aware loss using Sobel filters.
        
        Args:
            predicted (torch.Tensor): Predicted image [B, 3, H, W]
            target (torch.Tensor): Ground truth image [B, 3, H, W]
            
        Returns:
            torch.Tensor: Edge-aware loss value
        """
        # Extract edges
        pred_edges_x = F.conv2d(predicted, self.sobel_x, padding=1, groups=3)
        pred_edges_y = F.conv2d(predicted, self.sobel_y, padding=1, groups=3)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        
        # Compute edge magnitude
        pred_edges = torch.sqrt(pred_edges_x**2 + pred_edges_y**2 + 1e-8)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-8)
        
        # Compute L1 loss on edges
        return self.l1_loss(pred_edges, target_edges)
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the combined loss.
        
        Args:
            predicted (torch.Tensor): Predicted image [B, 3, H, W]
            target (torch.Tensor): Ground truth image [B, 3, H, W]
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with total loss and components
        """
        # Compute component losses
        l1 = self.l1_loss(predicted, target)
        ssim = self._ssim_loss(predicted, target)
        perceptual = self.perceptual_loss(predicted, target)
        edge = self._edge_loss(predicted, target)
        
        # Compute weighted sum
        total_loss = (
            self.l1_weight * l1 +
            self.ssim_weight * ssim +
            self.perceptual_weight * perceptual +
            self.edge_weight * edge
        )
        
        # Return all components for logging
        return {
            'loss': total_loss,
            'l1_loss': l1,
            'ssim_loss': ssim,
            'perceptual_loss': perceptual,
            'edge_loss': edge
        }


class MuralTrainer:
    """
    Trainer for the Dun Huang Mural restoration model.
    
    This class handles the complete training loop, including:
    - Data loading and batching
    - Optimization and learning rate scheduling
    - Validation and checkpoint management
    - Logging and visualization
    
    Attributes:
        model (MuralRestorationModel): The mural restoration model
        optimizer (optim.Optimizer): Optimization algorithm
        scheduler (optim.lr_scheduler._LRScheduler): Learning rate scheduler
        criterion (MuralLoss): Loss function
        args (argparse.Namespace): Training arguments
    """
    
    def __init__(
        self,
        args: argparse.Namespace,
        model: Optional[MuralRestorationModel] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            args (argparse.Namespace): Training arguments
            model (Optional[MuralRestorationModel]): Pre-initialized model (if None, creates new)
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        
        # Initialize or load model
        if model is None:
            self.model = MuralRestorationModel(
                pretrained_encoder=True,
                use_skip_connections=args.use_skip_connections,
                encoder_stages_to_unfreeze=args.unfreeze_encoder_stages,
                optimize_memory=args.optimize_memory
            )
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = MuralLoss(
            l1_weight=args.l1_weight,
            ssim_weight=args.ssim_weight,
            perceptual_weight=args.perceptual_weight,
            edge_weight=args.edge_weight
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2)
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._init_data_loaders()
        
        # Initialize training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_psnr = 0.0
        
        # Enable automatic mixed precision if specified
        self.use_amp = args.use_amp and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
        
        # Load checkpoint if provided
        if args.resume:
            self._load_checkpoint(args.resume)
        
        # Print model information
        trainable_params, total_params = self.model.get_trainable_parameters()
        print(f"Model initialized with {trainable_params:,}/{total_params:,} trainable parameters "
              f"({trainable_params/total_params*100:.2f}%)")
        
        # Enable memory efficient mode if specified
        if args.optimize_memory:
            self.model.enable_memory_efficient_mode()
            print("Memory-efficient mode enabled")
    
    def _init_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Initialize data loaders for training and validation.
        
        Returns:
            Tuple[DataLoader, DataLoader]: Training and validation data loaders
        """
        # Create datasets
        train_dataset = MuralDataset(
            data_path=self.args.data_path,
            mode='train',
            image_size=self.args.image_size,
            use_augmentation=True
        )
        
        val_dataset = MuralDataset(
            data_path=self.args.data_path,
            mode='val',
            image_size=self.args.image_size,
            use_augmentation=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_psnr': self.best_val_psnr
        }
        
        # Save scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save as best if applicable
        if is_best:
            best_path = os.path.join(
                self.args.output_dir, 'checkpoints', 'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            print(f"Saved best model with PSNR: {self.best_val_psnr:.2f} dB")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model and training state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state if available and using mixed precision
        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
        
        print(f"Resuming from epoch {self.current_epoch}")
    
    def _compute_psnr(self, predicted: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            predicted (torch.Tensor): Predicted image in [-1, 1] range
            target (torch.Tensor): Ground truth image in [-1, 1] range
            
        Returns:
            float: PSNR value in dB
        """
        # Convert from [-1, 1] to [0, 1]
        predicted = (predicted + 1) / 2
        target = (target + 1) / 2
        
        # Compute MSE
        mse = F.mse_loss(predicted, target).item()
        
        # Compute PSNR
        if mse == 0:
            return float('inf')
        
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def _save_samples(self, epoch: int, damaged: torch.Tensor, 
                      predicted: torch.Tensor, target: torch.Tensor) -> None:
        """
        Save sample restoration results.
        
        Args:
            epoch (int): Current epoch
            damaged (torch.Tensor): Damaged input images
            predicted (torch.Tensor): Predicted restored images
            target (torch.Tensor): Ground truth images
        """
        # Create sample grid
        sample_count = min(4, damaged.size(0))
        damaged = damaged[:sample_count]
        predicted = predicted[:sample_count]
        target = target[:sample_count]
        
        # Convert from [-1, 1] to [0, 1]
        damaged = (damaged + 1) / 2
        predicted = (predicted + 1) / 2
        target = (target + 1) / 2
        
        # Create comparison rows
        rows = []
        for i in range(sample_count):
            row = torch.cat([damaged[i], predicted[i], target[i]], dim=2)
            rows.append(row)
        
        # Combine rows
        grid = torch.cat(rows, dim=1)
        
        # Save image
        output_path = os.path.join(
            self.args.output_dir, 'samples', f'epoch_{epoch}.png'
        )
        save_image(grid, output_path)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            Dict[str, float]: Dictionary with training metrics
        """
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'l1_loss': 0.0,
            'ssim_loss': 0.0,
            'perceptual_loss': 0.0,
            'edge_loss': 0.0,
            'psnr': 0.0
        }
        
        start_time = time.time()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            damaged = batch['damaged'].to(self.device)
            gt = batch['gt'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predicted = self.model(damaged)
                    loss_dict = self.criterion(predicted, gt)
                
                # Backward pass with scaler
                self.scaler.scale(loss_dict['loss']).backward()
                
                # Clip gradients
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                predicted = self.model(damaged)
                loss_dict = self.criterion(predicted, gt)
                
                # Backward pass
                loss_dict['loss'].backward()
                
                # Clip gradients
                if self.args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                # Update weights
                self.optimizer.step()
            
            # Compute PSNR
            psnr = self._compute_psnr(predicted.detach(), gt)
            
            # Update metrics
            for k, v in loss_dict.items():
                epoch_metrics[k] += v.item()
            epoch_metrics['psnr'] += psnr
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss_dict['loss'].item(),
                'psnr': psnr
            })
        
        # Compute averages
        num_batches = len(self.train_loader)
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches
        
        # Log metrics
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{self.args.epochs} - "
              f"Train loss: {epoch_metrics['loss']:.4f}, "
              f"PSNR: {epoch_metrics['psnr']:.2f} dB, "
              f"Time: {elapsed:.2f}s")
        
        return epoch_metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            Dict[str, float]: Dictionary with validation metrics
        """
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'l1_loss': 0.0,
            'ssim_loss': 0.0,
            'perceptual_loss': 0.0,
            'edge_loss': 0.0,
            'psnr': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                # Move data to device
                damaged = batch['damaged'].to(self.device)
                gt = batch['gt'].to(self.device)
                
                # Forward pass with mixed precision if enabled
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        predicted = self.model(damaged)
                        loss_dict = self.criterion(predicted, gt)
                else:
                    predicted = self.model(damaged)
                    loss_dict = self.criterion(predicted, gt)
                
                # Compute PSNR
                psnr = self._compute_psnr(predicted, gt)
                
                # Update metrics
                for k, v in loss_dict.items():
                    val_metrics[k] += v.item()
                val_metrics['psnr'] += psnr
                
                # Save samples from the first batch
                if batch_idx == 0:
                    self._save_samples(epoch, damaged, predicted, gt)
        
        # Compute averages
        num_batches = len(self.val_loader)
        for k in val_metrics:
            val_metrics[k] /= num_batches
        
        # Log metrics
        print(f"Validation - "
              f"Loss: {val_metrics['loss']:.4f}, "
              f"PSNR: {val_metrics['psnr']:.2f} dB")
        
        # Update best metrics
        is_best = False
        if val_metrics['psnr'] > self.best_val_psnr:
            self.best_val_psnr = val_metrics['psnr']
            self.best_val_loss = val_metrics['loss']
            is_best = True
        
        # Save checkpoint
        self._save_checkpoint(epoch, is_best=is_best)
        
        return val_metrics
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Returns:
            Dict[str, List[float]]: Dictionary with training and validation metrics
        """
        # Initialize metrics history
        history = {
            'train_loss': [],
            'train_psnr': [],
            'val_loss': [],
            'val_psnr': []
        }
        
        # Train for specified number of epochs
        for epoch in range(self.current_epoch, self.args.epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])
            history['train_psnr'].append(train_metrics['psnr'])
            
            # Validate
            val_metrics = self.validate(epoch)
            history['val_loss'].append(val_metrics['loss'])
            history['val_psnr'].append(val_metrics['psnr'])
            
            # Update learning rate
            self.scheduler.step()
            
            # Check for early stopping
            if self.args.patience > 0:
                if len(history['val_psnr']) > self.args.patience:
                    if all(history['val_psnr'][-i-1] >= history['val_psnr'][-i] 
                           for i in range(1, self.args.patience + 1)):
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        print(f"Training completed with best PSNR: {self.best_val_psnr:.2f} dB")
        return history


def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Dun Huang Mural Restoration Model")
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--image_size', type=int, default=256, help="Input image size")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of data loading workers")
    
    # Model arguments
    parser.add_argument('--use_skip_connections', action='store_true', help="Use skip connections")
    parser.add_argument('--optimize_memory', action='store_true', help="Enable memory optimizations")
    parser.add_argument('--unfreeze_encoder_stages', nargs='+', default=['stage1'], 
                      choices=['stage1', 'stage2', 'stage3', 'stage4'], 
                      help="Encoder stages to unfreeze")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for Adam optimizer")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for Adam optimizer")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience (0 to disable)")
    parser.add_argument('--use_amp', action='store_true', help="Use automatic mixed precision")
    
    # Loss function arguments
    parser.add_argument('--l1_weight', type=float, default=1.0, help="Weight for L1 loss")
    parser.add_argument('--ssim_weight', type=float, default=0.5, help="Weight for SSIM loss")
    parser.add_argument('--perceptual_weight', type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument('--edge_weight', type=float, default=0.2, help="Weight for edge-aware loss")
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./output', help="Output directory")
    parser.add_argument('--resume', type=str, default='', help="Path to checkpoint for resuming training")
    
    args = parser.parse_args()
    return args


def main():
    """Main training function."""
    # Parse arguments
    args = get_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Initialize trainer
    trainer = MuralTrainer(args)
    
    # Train model
    history = trainer.train()
    
    # Save final results
    final_checkpoint_path = os.path.join(
        args.output_dir, 'checkpoints', 'final_model.pth'
    )
    torch.save(trainer.model.state_dict(), final_checkpoint_path)
    
    print(f"Training completed with best PSNR: {trainer.best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
