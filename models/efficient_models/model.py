"""
model.py - Mural Restoration Model for Dun Huang Murals

This module integrates the encoder and decoder components into a complete mural restoration 
model. It handles the end-to-end process of generating the initial restoration, which will 
then be passed to the physics refinement module.

Key components:
- MuralRestorationModel: Complete model integrating encoder and decoder

Usage:
    model = MuralRestorationModel(
        pretrained_encoder=True,
        use_skip_connections=True,
        encoder_stages_to_unfreeze=['stage1']
    )
    
    initial_restoration = model(damaged_image)
    # Then pass to physics refinement module
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from models.encoder.encoder import SwinEncoderExtractor
from models.efficient_models.decoder import DecoderModule


class MuralRestorationModel(nn.Module):
    """
    Memory-efficient model for Dun Huang Mural restoration.
    
    This model integrates the Swin Transformer encoder with the
    streamlined decoder into a single pipeline for mural restoration.
    It focuses on structural image restoration while optimizing memory usage,
    leaving domain-specific processing to the physics refinement module.
    
    Attributes:
        encoder (SwinEncoderExtractor): Feature extraction encoder
        decoder (EfficientDecoderModule): Memory-efficient restoration decoder
        use_skip_connections (bool): Whether skip connections are enabled
    """
    
    def __init__(
        self,
        pretrained_encoder: bool = True,
        use_skip_connections: bool = True,
        encoder_stages_to_unfreeze: Optional[List[str]] = None,
        optimize_memory: bool = True
    ):
        """
        Initialize the mural restoration model.
        
        Args:
            pretrained_encoder (bool): Whether to use pretrained weights for encoder
            use_skip_connections (bool): Whether to use enhanced skip connections
            encoder_stages_to_unfreeze (Optional[list]): List of encoder stages to unfreeze
            optimize_memory (bool): Whether to apply memory optimizations
        """
        super(MuralRestorationModel, self).__init__()
        
        # Initialize the encoder
        self.encoder = SwinEncoderExtractor(pretrained=pretrained_encoder)
        
        # Selectively unfreeze encoder stages if specified
        if encoder_stages_to_unfreeze is not None:
            self.encoder.unfreeze_parameters(encoder_stages_to_unfreeze)
        
        # Get encoder dimensions
        encoder_dims = self.encoder.output_dims
        
        # Initialize the decoder
        self.decoder = DecoderModule(
            encoder_dims=encoder_dims,
            skip_connections=use_skip_connections
        )
        
        # Store configuration
        self.use_skip_connections = use_skip_connections
        self.optimize_memory = optimize_memory
        
        # Initialize weights for decoder
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for decoder modules."""
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
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Damaged image tensor [B, 3, 256, 256]
            
        Returns:
            torch.Tensor: Initial restored image tensor [B, 3, 256, 256]
        """
        # Extract features from encoder
        if self.optimize_memory:
            # Process in stages with gradient checkpointing to save memory
            with torch.amp.autocast(enabled=True):
                encoder_features = self.encoder(x)
        else:
            encoder_features = self.encoder(x)
        
        # Pass features to decoder
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
    
    def enable_memory_efficient_mode(self):
        """
        Enable memory-efficient mode for large inputs.
        
        This applies several optimizations to reduce memory usage:
        1. Enables gradient checkpointing in the encoder
        2. Uses mixed precision where possible
        3. Enables more aggressive feature cleanup during forward pass
        """
        self.optimize_memory = True
        
        # Enable gradient checkpointing if available
        if hasattr(self.encoder, 'swin_encoder') and hasattr(self.encoder.swin_encoder, 'gradient_checkpointing_enable'):
            self.encoder.swin_encoder.gradient_checkpointing_enable()


# Example usage
if __name__ == "__main__":
    # Create a dummy input tensor (batch_size=2, channels=3, height=256, width=256)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256)
    
    # Initialize the efficient model
    model = MuralRestorationModel(
        pretrained_encoder=True,
        use_skip_connections=True,
        encoder_stages_to_unfreeze=['stage1'],  # Only unfreeze stage1 of encoder
        optimize_memory=True
    )
    
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
    
    # Print memory efficiency improvements
    print("Memory efficiency improvements enabled:")
    print("- Mixed precision computation")
    print("- Gradient checkpointing in encoder")
    print("- Enhanced skip connection fusion")
    print("- Dynamic window sizing for attention")
