"""
model.py - High-Performance Mural Restoration Model for Dun Huang Murals

This module integrates the encoder and high-performance decoder into a complete
mural restoration model optimized for HPC training. It allocates more parameters
to critical components while maintaining efficiency in less important areas.

Key components:
- HighPerformanceMuralModel: Complete model integrating encoder and enhanced decoder
- Selective parameter allocation based on feature importance
- Mixed precision and gradient checkpointing options for HPC training

Usage:
    model = HighPerformanceMuralModel(
        pretrained_encoder=True,
        use_skip_connections=True,
        encoder_stages_to_unfreeze=['stage1'],
        high_capacity_mode='balanced'
    )
    
    initial_restoration = model(damaged_image)
    # Then pass to physics refinement module
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from encoder import SwinEncoderExtractor
from decoder import HighPerformanceDecoderModule


class HighPerformanceMuralModel(nn.Module):
    """
    High-performance model for Dun Huang Mural restoration.
    
    This model integrates the Swin Transformer encoder with the
    enhanced decoder into a single pipeline optimized for HPC training,
    allocating more parameters to critical components.
    
    Attributes:
        encoder (SwinEncoderExtractor): Feature extraction encoder
        decoder (HighPerformanceDecoderModule): Enhanced restoration decoder
        use_skip_connections (bool): Whether skip connections are enabled
        use_mixed_precision (bool): Whether to use mixed precision for inference
    """
    
    def __init__(
        self,
        pretrained_encoder: bool = True,
        use_skip_connections: bool = True,
        encoder_stages_to_unfreeze: Optional[List[str]] = None,
        high_capacity_mode: str = 'balanced',
        use_mixed_precision: bool = True
    ):
        """
        Initialize the high-performance mural restoration model.
        
        Args:
            pretrained_encoder (bool): Whether to use pretrained weights for encoder
            use_skip_connections (bool): Whether to use enhanced skip connections
            encoder_stages_to_unfreeze (Optional[list]): List of encoder stages to unfreeze
            high_capacity_mode (str): Where to allocate more parameters:
                - 'early': More parameters in early stages for fine details
                - 'deep': More parameters in deep stages for semantic information
                - 'balanced': Distributed parameter allocation
            use_mixed_precision (bool): Whether to use mixed precision
        """
        super(HighPerformanceMuralModel, self).__init__()
        
        # Initialize the encoder
        self.encoder = SwinEncoderExtractor(pretrained=pretrained_encoder)
        
        # Selectively unfreeze encoder stages if specified
        if encoder_stages_to_unfreeze is not None:
            self.encoder.unfreeze_parameters(encoder_stages_to_unfreeze)
        
        # Get encoder dimensions
        encoder_dims = self.encoder.output_dims
        
        # Initialize the enhanced decoder
        self.decoder = HighPerformanceDecoderModule(
            encoder_dims=encoder_dims,
            skip_connections=use_skip_connections,
            high_capacity_mode=high_capacity_mode
        )
        
        # Store configuration
        self.use_skip_connections = use_skip_connections
        self.use_mixed_precision = use_mixed_precision
        
        # Initialize weights for decoder
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for decoder modules with enhanced initialization."""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the high-performance model.
        
        Args:
            x (torch.Tensor): Damaged image tensor [B, 3, 256, 256]
            
        Returns:
            torch.Tensor: Initial restored image tensor [B, 3, 256, 256]
        """
        # Process with optional mixed precision for HPC
        if self.use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=True):
                # Extract features from encoder
                encoder_features = self.encoder(x)
                
                # Pass features to enhanced decoder
                restored_image = self.decoder(encoder_features)
        else:
            # Extract features from encoder
            encoder_features = self.encoder(x)
            
            # Pass features to enhanced decoder
            restored_image = self.decoder(encoder_features)
        
        return restored_image
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """
        Get the number of trainable and total parameters.
        
        Returns:
            Tuple[int, int]: Number of trainable parameters and total parameters
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return trainable_params, total_params
    
    def get_param_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of parameters across model components.
        
        Returns:
            Dict[str, int]: Number of parameters in each model component
        """
        # Get encoder parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        
        # Get decoder stage parameters
        stage1_params = sum(p.numel() for name, p in self.decoder.named_parameters() 
                           if name.startswith(('stage1', 'compress_stage3', 'compress_stage4')))
        
        stage2_params = sum(p.numel() for name, p in self.decoder.named_parameters() 
                           if name.startswith(('stage2', 'compress_stage2')))
        
        stage3_params = sum(p.numel() for name, p in self.decoder.named_parameters() 
                           if name.startswith('stage3'))
        
        stage4_params = sum(p.numel() for name, p in self.decoder.named_parameters() 
                           if name.startswith('stage4'))
        
        output_params = sum(p.numel() for name, p in self.decoder.named_parameters() 
                           if name.startswith('output_proj'))
        
        return {
            'encoder': encoder_params,
            'decoder_stage1': stage1_params,
            'decoder_stage2': stage2_params,
            'decoder_stage3': stage3_params,
            'decoder_stage4': stage4_params,
            'decoder_output': output_params
        }
    
    def setup_for_hpc_training(self):
        """
        Configure the model for optimal HPC training.
        
        Enables gradient checkpointing, mixed precision, and other
        optimizations for training on HPC clusters.
        """
        # Enable mixed precision
        self.use_mixed_precision = True
        
        # Enable gradient checkpointing in encoder if available
        if hasattr(self.encoder, 'swin_encoder') and hasattr(self.encoder.swin_encoder, 'gradient_checkpointing_enable'):
            self.encoder.swin_encoder.gradient_checkpointing_enable()


# Example usage
if __name__ == "__main__":
    # Create a dummy input tensor (batch_size=2, channels=3, height=256, width=256)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    
    # Initialize the high-performance model
    model = HighPerformanceMuralModel(
        pretrained_encoder=True,
        use_skip_connections=True,
        encoder_stages_to_unfreeze=['stage1', 'stage2'],  # Unfreeze early stages
        high_capacity_mode='balanced',
        use_mixed_precision=True
    )
    
    # Configure for HPC training
    model.setup_for_hpc_training()
    
    # Forward pass
    restored_image = model(dummy_input)
    
    # Print output shape
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {restored_image.shape}")
    
    # Print parameter counts
    trainable_params, total_params = model.get_trainable_parameters()
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {trainable_params / total_params * 100:.2f}%")
    
    # Print parameter distribution
    param_dist = model.get_param_distribution()
    for component, params in param_dist.items():
        print(f"{component}: {params:,} parameters ({params / total_params * 100:.2f}%)")
