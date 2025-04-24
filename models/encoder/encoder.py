"""
encoder.py - Swin Transformer v2 Base Encoder for Dun Huang Mural Restoration

This module implements the encoder part of the mural restoration architecture using 
a pre-trained Swin Transformer v2 Base model. The encoder extracts multi-scale features
from damaged mural images at four different resolution levels.

Key components:
- SwinEncoderExtractor: Wrapper for Swin-v2-Base that extracts multi-level features
- Feature scale management: Handles the progressive downsampling and feature extraction
- Output feature dimensions:
  * Stage 1: 64×64×96   (fine details, brushstrokes, textures)
  * Stage 2: 32×32×192  (medium structures, patterns)
  * Stage 3: 16×16×384  (semantic information)
  * Stage 4: 8×8×768    (global context)

Usage:
    encoder = SwinEncoderExtractor(pretrained=True)
    features = encoder(damaged_image)  # Returns dict with multi-scale features
"""

import torch
import torch.nn as nn
from torchvision.models import swin_v2_b
from torchvision.models._utils import IntermediateLayerGetter
from collections import OrderedDict
from typing import Dict, List, Tuple

class SwinEncoderExtractor(nn.Module):
    """
    Swin Transformer v2 Base encoder for extracting multi-scale features from damaged mural images.
    
    The model uses a pre-trained Swin-v2-Base model and extracts features from intermediate
    layers to obtain a hierarchical representation at different scales and semantic levels.
    
    Attributes:
        swin_encoder (nn.Module): Modified Swin Transformer v2 Base model
        output_features (List[str]): Names of the output feature maps
        output_dims (Dict[str, int]): Dimensions of the output feature maps
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize the Swin Transformer v2 Base encoder.
        
        Args:
            pretrained (bool): Whether to use the pre-trained weights from ImageNet
        """
        super(SwinEncoderExtractor, self).__init__()
        
        # Define feature extraction points
        return_layers = {
            'features.0': 'stage1',    # After patch partition and first Swin blocks (64×64×96)
            'features.1': 'stage2',    # After first patch merging and second Swin blocks (32×32×192)
            'features.2': 'stage3',    # After second patch merging and third Swin blocks (16×16×384)
            'features.3': 'stage4'     # After third patch merging and fourth Swin blocks (8×8×768)
        }
        
        # Load pre-trained Swin Transformer v2 Base model
        base_model = swin_v2_b(pretrained=pretrained)
        
        # Extract just the features part (without the classifier)
        self.swin_encoder = IntermediateLayerGetter(base_model, return_layers=return_layers)
        
        # Store output feature names and dimensions for reference
        self.output_features = ['stage1', 'stage2', 'stage3', 'stage4']
        self.output_dims = {
            'stage1': 96,
            'stage2': 192,
            'stage3': 384,
            'stage4': 768
        }
        
        # Freeze the encoder parameters if using pre-trained model
        if pretrained:
            self._freeze_parameters()
    
    def _freeze_parameters(self):
        """Freeze all parameters of the encoder to use it as a feature extractor only."""
        for param in self.swin_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_parameters(self, stages_to_unfreeze: List[str] = None):
        """
        Selectively unfreeze parameters of specific stages for fine-tuning.
        
        Args:
            stages_to_unfreeze (List[str]): List of stage names to unfreeze ('stage1', 'stage2', etc.)
                                           If None, all stages remain frozen.
        """
        if stages_to_unfreeze is None:
            return
        
        for name, param in self.swin_encoder.named_parameters():
            for stage in stages_to_unfreeze:
                if stage in name:
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from the input image.
        
        Args:
            x (torch.Tensor): Input image tensor of shape [B, 3, 256, 256]
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of feature maps at different scales:
                - 'stage1': [B, 96, 64, 64]
                - 'stage2': [B, 192, 32, 32]
                - 'stage3': [B, 384, 16, 16]
                - 'stage4': [B, 768, 8, 8]
        """
        # Ensure input has the correct shape
        assert x.shape[2] == 256 and x.shape[3] == 256, f"Input image must be 256×256, got {x.shape[2]}×{x.shape[3]}"
        
        # Extract features from the Swin Transformer
        features = self.swin_encoder(x)
        
        return features

    def get_output_dims(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get the output dimensions of each feature map for a 256×256 input image.
        
        Returns:
            Dict[str, Tuple[int, int, int]]: Dictionary mapping stage names to (channels, height, width)
        """
        return {
            'stage1': (96, 64, 64),
            'stage2': (192, 32, 32),
            'stage3': (384, 16, 16),
            'stage4': (768, 8, 8)
        }


# Example usage
if __name__ == "__main__":
    # Create a dummy input tensor (batch_size=1, channels=3, height=256, width=256)
    dummy_input = torch.randn(1, 3, 256, 256)
    
    # Initialize the encoder
    encoder = SwinEncoderExtractor(pretrained=True)
    
    # Print model information
    print(f"Encoder output features: {encoder.output_features}")
    print(f"Encoder output dimensions: {encoder.output_dims}")
    
    # Extract features
    features = encoder(dummy_input)
    
    # Print feature shapes
    for stage_name, feature_map in features.items():
        print(f"{stage_name} shape: {feature_map.shape}")
