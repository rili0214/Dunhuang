"""
decoder.py - Memory-Efficient Decoder for Dun Huang Mural Restoration

Key components:
- ChannelCompression: Reduces channel dimensions
- SkipFusion: Enhanced skip connection 
- AdaptiveCSABlock: Generic cross-scale attention with dynamic window sizes
- UpsampleBlock: Various upsampling strategies with residual refinement
- EfficientDecoderModule: Streamlined decoder implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


def get_dynamic_window_size(resolution: Tuple[int, int]) -> int:
    """
    Calculate appropriate window size based on feature resolution.
    """
    avg_size = (resolution[0] + resolution[1]) // 2
    
    if avg_size <= 8:
        return 8 
    elif avg_size <= 16:
        return 6 
    elif avg_size <= 32:
        return 4 
    else:
        return 2 


class ChannelCompression(nn.Module):
    """
    Compresses feature channels while preserving important information.
    """
    def __init__(self, in_channels: int, out_channels: int, use_activation: bool = True):
        super(ChannelCompression, self).__init__()
        
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True) if use_activation else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(x)


class SkipFusion(nn.Module):
    """
    Enhanced skip connection with concatenation and 1×1 convolution.
    """
    
    def __init__(self, decoder_channels: int, encoder_channels: int, out_channels: int):
        super(SkipFusion, self).__init__()
        
        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_channels + encoder_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, decoder_feat: torch.Tensor, encoder_feat: torch.Tensor) -> torch.Tensor:
        # Align encoder features to decoder resolution
        if decoder_feat.shape[2:] != encoder_feat.shape[2:]:
            aligned_encoder = F.interpolate(
                encoder_feat, 
                size=decoder_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            aligned_encoder = encoder_feat
        
        # Concatenate and fuse
        concat_features = torch.cat([decoder_feat, aligned_encoder], dim=1)
        fused_features = self.fusion(concat_features)
        
        return fused_features


class ResolutionAligner(nn.Module):
    """
    Aligns and fuses features from different resolutions.
    """
    def __init__(
        self, 
        input_dims: Dict[str, int],
        target_channels: int,
        target_size: Tuple[int, int]
    ):
        """
        Initialize the resolution aligner.
        """
        super(ResolutionAligner, self).__init__()
        
        self.target_channels = target_channels
        self.target_size = target_size
        self.compressions = nn.ModuleDict({
            name: ChannelCompression(dims, target_channels)
            for name, dims in input_dims.items()
        })
        num_groups = min(4, len(input_dims))
        self.fusion = nn.Sequential(
            nn.Conv2d(
                target_channels * len(input_dims), 
                target_channels, 
                kernel_size=1, 
                groups=num_groups
            ),
            nn.InstanceNorm2d(target_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        aligned_features = []
        
        for name, feature in features.items():
            compressed = self.compressions[name](feature)
            if compressed.shape[2:] != self.target_size:
                compressed = F.interpolate(
                    compressed, size=self.target_size, mode='bilinear', align_corners=False
                )
            
            aligned_features.append(compressed)
            
        x = torch.cat(aligned_features, dim=1)
        x = self.fusion(x)
        
        return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention.
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
        
        # Get pair-wise relative position index (precomputed)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        
        # Calculate relative positions
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # QKV projection (grouped)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        
        # QKV computation
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scale query
        q = q * self.scale
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1))
        
        # Apply relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

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


class CSABlock(nn.Module):
    """
    Cross-Scale Attention Block.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super(CSABlock, self).__init__()
        
        # Normalization and MLP layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Windowed self-attention modules
        self.attn_modules = nn.ModuleDict({
            f'window_{size}': WindowAttention(
                dim=dim,
                window_size=size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop
            ) for size in [2, 4, 6, 8]  # Predefine all possible window sizes
        })
        
        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Store parameters for dynamic window attention
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = drop
    
    def _window_partition(self, x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple]:
        """
        Partition input feature into windows.
        """
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w

        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        
        return windows, (H, W, pad_h, pad_w)
    
    def _window_reverse(
        self, windows: torch.Tensor, window_size: int, H: int, W: int, pad_h: int, pad_w: int
    ) -> torch.Tensor:
        """
        Reverse window partitioning.
        """
        B = int(windows.shape[0] / ((H // window_size) * (W // window_size)))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H-pad_h, :W-pad_w, :]
            
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x
        
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

        window_size = get_dynamic_window_size((H, W))
        window_attn = self.attn_modules[f'window_{window_size}']
        
        # Apply window attention
        x_norm = self.norm1(x)
        windows, (H_pad, W_pad, pad_h, pad_w) = self._window_partition(x_norm, window_size)
        
        # Create window attention dynamically based on resolution
        window_attn = WindowAttention(
            dim=self.dim,
            window_size=window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop
        ).to(x.device)
        
        attn_windows = window_attn(windows)
        x_attn = self._window_reverse(attn_windows, window_size, H_pad, W_pad, pad_h, pad_w)
        
        # First residual connection
        x = x + x_attn
        
        # Feed-forward network with residual connection
        x = x + self.mlp(self.norm2(x))
        
        # Change back to BCHW format
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        
        return x


class UpsampleBlock(nn.Module):
    """
    Upsampling block with residual refinement.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = 'pixel_shuffle'
    ):
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
            # Pixel shuffle for upsampling (memory-efficient)
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
        
        # Residual refinement block
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        
        return x + self.refine(x)


class DecoderModule(nn.Module):
    """
    Decoder module for Dun Huang Mural restoration.
    """
    def __init__(
        self,
        encoder_dims: Dict[str, int],
        skip_connections: bool = True
    ):
        super(DecoderModule, self).__init__()
        
        self.skip_connections = skip_connections
        
        # Store input dimensions for reference
        self.encoder_dims = encoder_dims
        
        # Channel compression for deep features
        self.compress_stage3 = ChannelCompression(encoder_dims['stage3'], 256)
        self.compress_stage4 = ChannelCompression(encoder_dims['stage4'], 256)
        
        # Stage 1: Process deepest features (8×8)
        self.stage1_ra = ResolutionAligner(
            input_dims={'stage3': 256, 'stage4': 256},
            target_channels=256,
            target_size=(8, 8)
        )
        self.stage1_csa = CSABlock(
            dim=256,
            num_heads=8,
            mlp_ratio=4.0
        )
        self.stage1_up = UpsampleBlock(
            in_channels=256,
            out_channels=192,
            scale_factor=2,
            mode='transpose_conv'
        )
        
        # Skip fusion for stage 1
        self.skip_fusion1 = SkipFusion(
            decoder_channels=192,
            encoder_channels=encoder_dims['stage4'],
            out_channels=192
        )
        
        # Stage 2: Process mid-level features (16×16)
        self.compress_stage2 = ChannelCompression(encoder_dims['stage2'], 128)
        self.stage2_ra = ResolutionAligner(
            input_dims={'stage2': 128, 'stage3': 256, 'mid1': 192},
            target_channels=192,
            target_size=(16, 16)
        )
        self.stage2_csa = CSABlock(
            dim=192,
            num_heads=4
        )
        self.stage2_up = UpsampleBlock(
            in_channels=192,
            out_channels=96,
            scale_factor=2,
            mode='pixel_shuffle'
        )
        
        # Skip fusion for stage 2
        self.skip_fusion2 = SkipFusion(
            decoder_channels=96,
            encoder_channels=encoder_dims['stage3'],
            out_channels=96
        )
        
        # Stage 3: Process mid-to-early features (32×32)
        self.stage3_ra = ResolutionAligner(
            input_dims={'stage1': encoder_dims['stage1'], 'stage2': 128, 'mid2': 96},
            target_channels=96,
            target_size=(32, 32)
        )
        self.stage3_csa = CSABlock(
            dim=96,
            num_heads=4
        )
        self.stage3_up = UpsampleBlock(
            in_channels=96,
            out_channels=64,
            scale_factor=2,
            mode='pixel_shuffle'
        )
        
        # Skip fusion for stage 3
        self.skip_fusion3 = SkipFusion(
            decoder_channels=64,
            encoder_channels=encoder_dims['stage2'],
            out_channels=64
        )
        
        # Stage 4: Process fine details (64×64 to 256×256)
        self.stage4_ra = ResolutionAligner(
            input_dims={'stage1': encoder_dims['stage1'], 'mid3': 64},
            target_channels=64,
            target_size=(64, 64)
        )
        self.stage4_csa = CSABlock(
            dim=64,
            num_heads=4,
            mlp_ratio=2.0
        )
        self.stage4_up = UpsampleBlock(
            in_channels=64,
            out_channels=32,
            scale_factor=4,
            mode='learned_pixel_shuffle'
        )
        
        # Skip fusion for stage 4
        self.skip_fusion4 = SkipFusion(
            decoder_channels=32,
            encoder_channels=encoder_dims['stage1'],
            out_channels=32
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
        Forward pass through the efficient decoder.
        
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
        
        # Apply enhanced skip connection
        if self.skip_connections:
            stage1_out = self.skip_fusion1(stage1_up, encoder_features['stage4'])
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
        
        # Apply enhanced skip connection
        if self.skip_connections:
            stage2_out = self.skip_fusion2(stage2_up, encoder_features['stage3'])
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
        
        # Apply enhanced skip connection
        if self.skip_connections:
            stage3_out = self.skip_fusion3(stage3_up, encoder_features['stage2'])
        else:
            stage3_out = stage3_up
        
        # Stage 4: Process fine details (64×64)
        stage4_input = {
            'stage1': encoder_features['stage1'],
            'mid3': stage3_out
        }
        stage4_aligned = self.stage4_ra(stage4_input)
        stage4_attn = self.stage4_csa(stage4_aligned)
        stage4_up = self.stage4_up(stage4_attn)
        
        # Apply enhanced skip connection
        if self.skip_connections:
            stage4_out = self.skip_fusion4(stage4_up, encoder_features['stage1'])
        else:
            stage4_out = stage4_up
        
        # Output projection
        restored_image = self.output_proj(stage4_out)
        
        return restored_image
