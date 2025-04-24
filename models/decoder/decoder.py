"""
decoder.py - Cross-Scale Attention Decoder for Dun Huang Mural Restoration

This module implements the decoder part of the mural restoration architecture with
Cross-Scale Attention (CSA) blocks, dynamic window sizes, and skip connections.
The decoder progressively upsamples features while integrating multi-scale information
from the encoder.

Key components:
- ChannelCompression: Reduces channel dimensions while preserving information
- ResolutionAligner: Aligns features from different scales for fusion
- AdaptiveCSABlock: Cross-Scale Attention with dynamic window sizes
- UpsampleBlock: Various upsampling strategies (transposed conv, pixel shuffle)
- MuralDecoderModule: Complete decoder implementation

Architecture flow:
1. Compress deep features from encoder
2. Process through 4 progressive decoder stages with skip connections
3. Apply content and pigment-aware attention mechanisms
4. Output high-fidelity restored mural image

Usage:
    # Initialize with dimensions matching the encoder
    decoder = MuralDecoderModule(encoder_dims={
        'stage1': 96, 'stage2': 192, 'stage3': 384, 'stage4': 768
    })
    
    # Forward pass
    restored_image = decoder(encoder_features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class ChannelCompression(nn.Module):
    """
    Compresses feature channels while preserving important information.
    
    This module reduces the number of channels using a combination of
    1×1 convolution and instance normalization to maintain feature distinctiveness.
    
    Attributes:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        use_activation (bool): Whether to apply activation after compression
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_activation: bool = True):
        """
        Initialize the channel compression module.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            use_activation (bool): Whether to apply ReLU activation
        """
        super(ChannelCompression, self).__init__()
        
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True) if use_activation else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input feature channels.
        
        Args:
            x (torch.Tensor): Input features [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: Compressed features [B, out_channels, H, W]
        """
        return self.compress(x)


class ResolutionAligner(nn.Module):
    """
    Aligns and fuses features from different resolutions.
    
    This module handles multiple input features at different resolutions,
    aligning them to a target resolution and channel dimension before fusion.
    
    Attributes:
        target_channels (int): Target number of output channels
        target_size (Tuple[int, int]): Target spatial dimensions (H, W)
    """
    
    def __init__(
        self, 
        input_dims: Dict[str, int],
        target_channels: int,
        target_size: Tuple[int, int]
    ):
        """
        Initialize the resolution aligner.
        
        Args:
            input_dims (Dict[str, int]): Dictionary mapping feature names to channel dimensions
            target_channels (int): Target number of output channels
            target_size (Tuple[int, int]): Target output resolution (H, W)
        """
        super(ResolutionAligner, self).__init__()
        
        self.target_channels = target_channels
        self.target_size = target_size
        
        # Create channel compression modules for each input
        self.compressions = nn.ModuleDict({
            name: ChannelCompression(dims, target_channels)
            for name, dims in input_dims.items()
        })
        
        # Fusion convolution to combine aligned features
        self.fusion = nn.Sequential(
            nn.Conv2d(target_channels * len(input_dims), target_channels, kernel_size=1),
            nn.InstanceNorm2d(target_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Align and fuse multiple input features.
        
        Args:
            features (Dict[str, torch.Tensor]): Dictionary of feature tensors at different scales
            
        Returns:
            torch.Tensor: Fused features at target resolution and channels [B, target_channels, H, W]
        """
        aligned_features = []
        
        for name, feature in features.items():
            # Compress channels
            compressed = self.compressions[name](feature)
            
            # Resize to target size if needed
            if compressed.shape[2:] != self.target_size:
                compressed = F.interpolate(
                    compressed, size=self.target_size, mode='bilinear', align_corners=False
                )
            
            aligned_features.append(compressed)
        
        # Concatenate and fuse
        x = torch.cat(aligned_features, dim=1)
        x = self.fusion(x)
        
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention module with relative position encoding.
    
    This is a modified version of the attention mechanism from the Swin Transformer,
    adapted for the specific needs of mural restoration with dynamic window sizes.
    
    Attributes:
        dim (int): Input feature dimension
        window_size (int): Size of the attention window
        num_heads (int): Number of attention heads
        qkv (nn.Linear): Linear projection for query, key, value
        proj (nn.Linear): Linear projection for output
        relative_position_bias_table (nn.Parameter): Relative position bias parameters
    """
    
    def __init__(
        self, 
        dim: int, 
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        """
        Initialize window attention module.
        
        Args:
            dim (int): Input feature dimension
            window_size (int): Size of the attention window
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to use bias in QKV projection
            attn_drop (float): Dropout rate for attention matrix
            proj_drop (float): Dropout rate for output projection
        """
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias parameters
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size*window_size]
        
        # Calculate relative positions
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Projections for query, key, value and output
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias table
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute windowed self-attention.
        
        Args:
            x (torch.Tensor): Input features with shape [num_windows*B, window_size*window_size, C]
            mask (Optional[torch.Tensor]): Attention mask with shape [num_windows, window_size*window_size, window_size*window_size]
                
        Returns:
            torch.Tensor: Attention output with shape [num_windows*B, window_size*window_size, C]
        """
        B_, N, C = x.shape
        
        # Project query, key, value
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, num_heads, N, C//num_heads]
        
        # Compute attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]
        
        # Apply relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )  # [window_size*window_size, window_size*window_size, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [num_heads, window_size*window_size, window_size*window_size]
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Project back to original dimension
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AdaptiveCSABlock(nn.Module):
    """
    Adaptive Cross-Scale Attention Block with dynamic window size.
    
    This block implements self-attention within windows, with the window size
    adapted to the feature resolution. It also includes a gating mechanism
    to control feature flow and a feed-forward network for transformation.
    
    Attributes:
        dim (int): Input feature dimension
        window_size (int): Size of attention window
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dimension to input dimension
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        content_aware: bool = False
    ):
        """
        Initialize the adaptive CSA block.
        
        Args:
            dim (int): Input feature dimension
            window_size (int): Size of attention window
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of MLP hidden dimension to input dimension
            qkv_bias (bool): Whether to use bias in attention QKV projection
            drop (float): Dropout rate for MLP
            attn_drop (float): Dropout rate for attention
            content_aware (bool): Whether to use content-aware gating
        """
        super(AdaptiveCSABlock, self).__init__()
        
        # Normalization and attention layers
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Content-aware gating mechanism (optional)
        self.content_aware = content_aware
        if content_aware:
            self.gate = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        
        # Window size and input dimension
        self.window_size = window_size
        self.dim = dim
    
    def _window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Partition input feature into windows.
        
        Args:
            x (torch.Tensor): Input feature of shape [B, H, W, C]
            
        Returns:
            torch.Tensor: Windows of shape [num_windows*B, window_size, window_size, C]
        """
        B, H, W, C = x.shape
        
        # Pad features if needed to ensure divisibility by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        # Reshape into windows
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        
        return windows, (H, W, pad_h, pad_w)
    
    def _window_reverse(self, windows: torch.Tensor, H: int, W: int, pad_h: int, pad_w: int) -> torch.Tensor:
        """
        Reverse window partitioning.
        
        Args:
            windows (torch.Tensor): Windows of shape [num_windows*B, window_size*window_size, C]
            H (int): Padded height
            W (int): Padded width
            pad_h (int): Padding height
            pad_w (int): Padding width
            
        Returns:
            torch.Tensor: Reversed feature of shape [B, H-pad_h, W-pad_w, C]
        """
        B = int(windows.shape[0] / ((H // self.window_size) * (W // self.window_size)))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        
        # Remove padding if needed
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H-pad_h, :W-pad_w, :]
            
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CSA block.
        
        Args:
            x (torch.Tensor): Input feature of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Processed feature of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x
        
        # Change from BCHW to BHWC for window attention
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        
        # Apply window attention
        x_norm = self.norm1(x)
        windows, (H_pad, W_pad, pad_h, pad_w) = self._window_partition(x_norm)
        attn_windows = self.attn(windows)
        x_attn = self._window_reverse(attn_windows, H_pad, W_pad, pad_h, pad_w)
        
        # Content-aware gating (if enabled)
        if self.content_aware:
            # Compute content-aware gate
            gate_value = self.gate(x_norm.reshape(-1, C)).view(B, H, W, C)
            x_attn = x_attn * gate_value
        
        # First residual connection
        x = x + x_attn
        
        # Feed-forward network with residual connection
        x = x + self.mlp(self.norm2(x))
        
        # Change back to BCHW format
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block with multiple upsampling strategies.
    
    This module handles upsampling of feature maps using either transposed convolution
    or pixel shuffle, with appropriate normalization and activation.
    
    Attributes:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        scale_factor (int): Upsampling scale factor
        mode (str): Upsampling mode ('transpose_conv' or 'pixel_shuffle')
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = 'pixel_shuffle'
    ):
        """
        Initialize the upsampling block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            scale_factor (int): Upsampling scale factor
            mode (str): Upsampling mode ('transpose_conv' or 'pixel_shuffle')
        """
        super(UpsampleBlock, self).__init__()
        
        self.mode = mode
        
        if mode == 'transpose_conv':
            # Transposed convolution for upsampling
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, 
                    kernel_size=3, stride=scale_factor, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif mode == 'pixel_shuffle':
            # Pixel shuffle for upsampling
            self.upsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * scale_factor * scale_factor, 
                    kernel_size=3, padding=1
                ),
                nn.PixelShuffle(scale_factor),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif mode == 'learned_pixel_shuffle':
            # Learned pixel shuffle with additional conv
            self.upsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * scale_factor * scale_factor, 
                    kernel_size=3, padding=1
                ),
                nn.PixelShuffle(scale_factor),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            raise ValueError(f"Unsupported upsampling mode: {mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample the input feature map.
        
        Args:
            x (torch.Tensor): Input feature map [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: Upsampled feature map [B, out_channels, H*scale_factor, W*scale_factor]
        """
        return self.upsample(x)


class PigmentAwareCSA(nn.Module):
    """
    Specialized Cross-Scale Attention block for pigment-specific processing.
    
    This module is designed to handle the unique characteristics of ancient mural pigments
    using depthwise separable convolutions and specialized attention mechanisms.
    
    Attributes:
        dim (int): Input feature dimension
        window_size (int): Size of attention window
        num_heads (int): Number of attention heads
        groups (int): Number of groups for depthwise convolution
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        groups: int = 4,
        drop: float = 0.0
    ):
        """
        Initialize the pigment-aware CSA block.
        
        Args:
            dim (int): Input feature dimension
            window_size (int): Size of attention window
            num_heads (int): Number of attention heads
            groups (int): Number of groups for depthwise convolution
            drop (float): Dropout rate
        """
        super(PigmentAwareCSA, self).__init__()
        
        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=groups)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Windowed attention with smaller dimension
        reduced_dim = dim // 2
        self.compress = nn.Conv2d(dim, reduced_dim, kernel_size=1)
        self.norm = nn.LayerNorm(reduced_dim)
        self.window_attn = WindowAttention(
            dim=reduced_dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=drop,
            proj_drop=drop
        )
        self.expand = nn.Conv2d(reduced_dim, dim, kernel_size=1)
        
        # Pigment-specific processing
        self.pigment_gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final processing
        self.final = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Store dimensions
        self.dim = dim
        self.window_size = window_size
    
    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        Partition input feature into windows (simplified version).
        
        Args:
            x (torch.Tensor): Input feature of shape [B, H, W, C]
            
        Returns:
            Tuple[torch.Tensor, Tuple]: Windows and padding information
        """
        B, H, W, C = x.shape
        
        # Pad features if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
        
        # Reshape into windows
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        
        return windows, (H, W, pad_h, pad_w)
    
    def _window_reverse(self, windows: torch.Tensor, H: int, W: int, pad_h: int, pad_w: int) -> torch.Tensor:
        """
        Reverse window partitioning (simplified version).
        
        Args:
            windows (torch.Tensor): Windows of shape [num_windows*B, window_size*window_size, C]
            H, W, pad_h, pad_w: Size and padding information
            
        Returns:
            torch.Tensor: Reversed feature of shape [B, H-pad_h, W-pad_w, C]
        """
        B = int(windows.shape[0] / ((H // self.window_size) * (W // self.window_size)))
        x = windows.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H-pad_h, :W-pad_w, :]
            
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the pigment-aware CSA block.
        
        Args:
            x (torch.Tensor): Input feature of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Processed feature of shape [B, C, H, W]
        """
        B, C, H, W = x.shape
        shortcut = x
        
        # Depthwise separable convolution path
        depthwise_features = self.pointwise(self.depthwise(x))
        
        # Window attention path with dimension reduction
        x_reduced = self.compress(x)
        x_reduced = x_reduced.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C//2]
        x_reduced = self.norm(x_reduced)
        
        # Apply windowed attention
        windows, (H_pad, W_pad, pad_h, pad_w) = self._window_partition(x_reduced)
        attn_windows = self.window_attn(windows)
        x_attn = self._window_reverse(attn_windows, H_pad, W_pad, pad_h, pad_w)
        
        # Back to BCHW format and expand channels
        x_attn = x_attn.permute(0, 3, 1, 2).contiguous()
        x_attn = self.expand(x_attn)
        
        # Pigment-specific gating
        gate = self.pigment_gate(x)
        x_gated = x_attn * gate + depthwise_features * (1 - gate)
        
        # Add residual and apply final processing
        x = shortcut + x_gated
        x = self.final(x)
        
        return x


class CulturalElementAttention(nn.Module):
    """
    Specialized attention module for cultural elements in murals.
    
    This module enhances attention to culturally significant details like
    faces, religious symbols, and narrative elements in the mural.
    
    Attributes:
        channels (int): Number of input channels
        reduction (int): Channel reduction factor for attention
        kernel_size (int): Size of spatial attention kernel
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        kernel_size: int = 7
    ):
        """
        Initialize the cultural element attention module.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Channel reduction factor for attention
            kernel_size (int): Size of spatial attention kernel
        """
        super(CulturalElementAttention, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cultural element attention to input features.
        
        Args:
            x (torch.Tensor): Input feature map [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention-enhanced feature map [B, C, H, W]
        """
        # Apply channel attention
        channel_attn = self.channel_attention(x)
        x_channel = x * channel_attn
        
        # Apply spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_attn = self.spatial_attention(spatial_input)
        
        return x_channel * spatial_attn


class MuralDecoderModule(nn.Module):
    """
    Complete decoder module for Dun Huang Mural restoration.
    
    This module implements the full decoder architecture with multi-stage
    upsampling, cross-scale attention, skip connections, and specialized
    processing for mural-specific features.
    
    Attributes:
        encoder_dims (Dict[str, int]): Dimensions of encoder feature maps
        skip_connections (bool): Whether to use skip connections
        use_dynamic_windows (bool): Whether to use dynamic window sizes
    """
    
    def __init__(
        self,
        encoder_dims: Dict[str, int],
        skip_connections: bool = True,
        use_dynamic_windows: bool = True
    ):
        """
        Initialize the mural decoder module.
        
        Args:
            encoder_dims (Dict[str, int]): Dictionary mapping encoder stage names to channel dimensions
            skip_connections (bool): Whether to use skip connections from encoder to decoder
            use_dynamic_windows (bool): Whether to adapt window sizes based on feature resolution
        """
        super(MuralDecoderModule, self).__init__()
        
        self.skip_connections = skip_connections
        self.use_dynamic_windows = use_dynamic_windows
        
        # Store input dimensions for reference
        self.encoder_dims = encoder_dims
        
        # Channel compression for deep features
        self.compress_stage3 = ChannelCompression(encoder_dims['stage3'], 256)
        self.compress_stage4 = ChannelCompression(encoder_dims['stage4'], 256)
        
        # Stage 1: Process deepest features (8×8)
        window_size_1 = 8 if use_dynamic_windows else 7
        self.stage1_ra = ResolutionAligner(
            input_dims={'stage3': 256, 'stage4': 256},
            target_channels=256,
            target_size=(8, 8)
        )
        self.stage1_csa = AdaptiveCSABlock(
            dim=256,
            window_size=window_size_1,
            num_heads=8,
            mlp_ratio=4.0,
            content_aware=True
        )
        self.stage1_up = UpsampleBlock(
            in_channels=256,
            out_channels=192,
            scale_factor=2,
            mode='transpose_conv'
        )
        
        # Stage 2: Process mid-level features (16×16)
        window_size_2 = 6 if use_dynamic_windows else 7
        self.compress_stage2 = ChannelCompression(encoder_dims['stage2'], 128)
        self.stage2_ra = ResolutionAligner(
            input_dims={'stage2': 128, 'stage3': 256, 'mid1': 192},
            target_channels=192,
            target_size=(16, 16)
        )
        self.stage2_csa = AdaptiveCSABlock(
            dim=192,
            window_size=window_size_2,
            num_heads=4,
            content_aware=True
        )
        self.stage2_up = UpsampleBlock(
            in_channels=192,
            out_channels=96,
            scale_factor=2,
            mode='pixel_shuffle'
        )
        
        # Stage 3: Process mid-to-early features (32×32)
        window_size_3 = 4 if use_dynamic_windows else 7
        self.stage3_ra = ResolutionAligner(
            input_dims={'stage1': encoder_dims['stage1'], 'stage2': 128, 'mid2': 96},
            target_channels=96,
            target_size=(32, 32)
        )
        self.stage3_csa = PigmentAwareCSA(
            dim=96,
            window_size=window_size_3,
            num_heads=4,
            groups=4
        )
        self.stage3_up = UpsampleBlock(
            in_channels=96,
            out_channels=64,
            scale_factor=2,
            mode='pixel_shuffle'
        )
        
        # Stage 4: Process fine details (64×64)
        window_size_4 = 2 if use_dynamic_windows else 5
        self.cultural_attn = CulturalElementAttention(encoder_dims['stage1'])
        self.stage4_ra = ResolutionAligner(
            input_dims={'stage1_cultural': encoder_dims['stage1'], 'mid3': 64},
            target_channels=64,
            target_size=(64, 64)
        )
        self.stage4_csa = AdaptiveCSABlock(
            dim=64,
            window_size=window_size_4,
            num_heads=4,
            mlp_ratio=2.0
        )
        self.stage4_up = UpsampleBlock(
            in_channels=64,
            out_channels=32,
            scale_factor=4,
            mode='learned_pixel_shuffle'
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, encoder_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            encoder_features (Dict[str, torch.Tensor]): Dictionary of encoder features
                - 'stage1': [B, 96, 64, 64]
                - 'stage2': [B, 192, 32, 32]
                - 'stage3': [B, 384, 16, 16]
                - 'stage4': [B, 768, 8, 8]
                
        Returns:
            torch.Tensor: Restored image [B, 3, 256, 256]
        """
        # Compress deep features
        stage3_compressed = self.compress_stage3(encoder_features['stage3'])
        stage4_compressed = self.compress_stage4(encoder_features['stage4'])
        
        # Stage 1: Process deepest features (8×8)
        stage1_input = {
            'stage3': stage3_compressed,
            'stage4': stage4_compressed
        }
        stage1_aligned = self.stage1_ra(stage1_input)
        stage1_attn = self.stage1_csa(stage1_aligned)
        stage1_up = self.stage1_up(stage1_attn)
        
        # Skip connection from stage4 if enabled
        if self.skip_connections:
            stage1_out = stage1_up + F.interpolate(
                encoder_features['stage4'], 
                size=stage1_up.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            stage1_out = stage1_up
        
        # Stage 2: Process mid-level features (16×16)
        stage2_compressed = self.compress_stage2(encoder_features['stage2'])
        stage2_input = {
            'stage2': stage2_compressed,
            'stage3': stage3_compressed,
            'mid1': stage1_out
        }
        stage2_aligned = self.stage2_ra(stage2_input)
        stage2_attn = self.stage2_csa(stage2_aligned)
        stage2_up = self.stage2_up(stage2_attn)
        
        # Skip connection from stage3 if enabled
        if self.skip_connections:
            stage2_out = stage2_up + F.interpolate(
                encoder_features['stage3'], 
                size=stage2_up.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            stage2_out = stage2_up
        
        # Stage 3: Process mid-to-early features (32×32)
        stage3_input = {
            'stage1': encoder_features['stage1'],
            'stage2': stage2_compressed,
            'mid2': stage2_out
        }
        stage3_aligned = self.stage3_ra(stage3_input)
        stage3_attn = self.stage3_csa(stage3_aligned)
        stage3_up = self.stage3_up(stage3_attn)
        
        # Skip connection from stage2 if enabled
        if self.skip_connections:
            stage3_out = stage3_up + F.interpolate(
                encoder_features['stage2'], 
                size=stage3_up.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            stage3_out = stage3_up
        
        # Stage 4: Process fine details (64×64)
        stage1_cultural = self.cultural_attn(encoder_features['stage1'])
        stage4_input = {
            'stage1_cultural': stage1_cultural,
            'mid3': stage3_out
        }
        stage4_aligned = self.stage4_ra(stage4_input)
        stage4_attn = self.stage4_csa(stage4_aligned)
        stage4_up = self.stage4_up(stage4_attn)
        
        # Skip connection from stage1 if enabled
        if self.skip_connections:
            stage4_out = stage4_up + F.interpolate(
                encoder_features['stage1'], 
                size=stage4_up.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            stage4_out = stage4_up
        
        # Output projection
        restored_image = self.output_proj(stage4_out)
        
        return restored_image


# Example usage
if __name__ == "__main__":
    # Create dummy encoder features
    batch_size = 2
    encoder_features = {
        'stage1': torch.randn(batch_size, 96, 64, 64),
        'stage2': torch.randn(batch_size, 192, 32, 32),
        'stage3': torch.randn(batch_size, 384, 16, 16),
        'stage4': torch.randn(batch_size, 768, 8, 8)
    }
    
    # Initialize the decoder
    decoder = MuralDecoderModule(
        encoder_dims={
            'stage1': 96,
            'stage2': 192,
            'stage3': 384,
            'stage4': 768
        },
        skip_connections=True,
        use_dynamic_windows=True
    )
    
    # Forward pass
    restored_image = decoder(encoder_features)
    
    # Print output shape
    print(f"Restored image shape: {restored_image.shape}")  # Should be [2, 3, 256, 256]
    
    # Print number of parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Number of parameters: {num_params:,}")
