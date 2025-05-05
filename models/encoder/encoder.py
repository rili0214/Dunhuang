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
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
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
        super().__init__()
        weights = Swin_V2_B_Weights.DEFAULT if pretrained else None
        base = swin_v2_b(weights=weights)

        # Patch embedding (features[0]) → [B, 96, 64, 64]
        self.patch_embed = base.features[0]
        # Four stages (features[1]…features[4])
        self.stages = nn.ModuleList([
            base.features[1],                   # stage1 blocks
            nn.Sequential(base.features[2],     # stage2 merge+blocks
                        base.features[3]),
            nn.Sequential(base.features[4],     # stage3 merge+blocks
                        base.features[5]),
            nn.Sequential(base.features[6],     # stage4 merge+blocks
                        base.features[7]),
        ])

        # Freeze all encoder parameters if pretrained
        if pretrained:
            self._freeze_parameters()

        # Metadata
        self.output_features = ['stage1', 'stage2', 'stage3', 'stage4']
        self.output_dims = {
            'stage1': 128,
            'stage2': 256,
            'stage3': 512,
            'stage4': 1024
        }

    
    def _freeze_parameters(self):
        """Freeze all parameters of the encoder to use it as a feature extractor only."""
        # patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # each stage
        for stage in self.stages:
            for param in stage.parameters():
                param.requires_grad = False
    
    def unfreeze_parameters(self, stages_to_unfreeze: List[str] = None):
        """
        Selectively unfreeze parameters of specific stages for fine-tuning.
        
        Args:
            stages_to_unfreeze (List[str]): List of stage names to unfreeze ('stage1', 'stage2', etc.)
                                           If None, all stages remain frozen.
        """
        if not stages_to_unfreeze:
            return

        name2idx = {'stage1':0, 'stage2':1, 'stage3':2, 'stage4':3}
        for stage_name in stages_to_unfreeze:
            idx = name2idx.get(stage_name)
            if idx is not None:
                for p in self.stages[idx].parameters():
                    p.requires_grad = True
    
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
        assert x.shape[2:] == (256, 256)
        x = self.patch_embed(x)  # -> [B, H, W, C] (BHWC)

        features: Dict[str, torch.Tensor] = {}
        for i, stage in enumerate(self.stages):
            x = stage(x)  # 仍然 BHWC

            # **在这里把 BHWC → BCHW，再保存**
            feat = x.permute(0, 3, 1, 2).contiguous()  
            features[f"stage{i+1}"] = feat

        return features


    def get_output_dims(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Get the output dimensions of each feature map for a 256×256 input image.
        
        Returns:
            Dict[str, Tuple[int, int, int]]: Dictionary mapping stage names to (channels, height, width)
        """
        return {
            'stage1': (128, 64, 64),
            'stage2': (256, 32, 32),
            'stage3': (512, 16, 16),
            'stage4': (1024, 8, 8),
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