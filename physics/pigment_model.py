import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
import json
import os

try:
    from chemparse import parse_formula     # pip install chemparse if not installed
except ImportError:
    # Fallback implementation if chemparse isn't available, should work as I believe
    def parse_formula(formula):
        """Simple chemical formula parser"""
        import re
        pattern = r'([A-Z][a-z]*)(\d*)'
        matches = re.findall(pattern, formula)
        result = {}
        for element, count in matches:
            result[element] = int(count) if count else 1
        return result

# ===== 1. Dunhuang Pigment Database =====
class DunhuangPigmentDB:
    """
    Database of historical pigments used in Dunhuang murals based on research paper data.
    Paper Link: https://skytyz.dha.ac.cn/CN/Y2022/V1/I1/47 
    Provides structured access to pigment properties including chemical composition,
    optical properties, aging characteristics,historical usage patterns, and particle size.
    """
    def __init__(self):
        self.pigments = self._init_pigment_data()
        self.color_to_pigment = self._init_color_mapping()
        self.pigment_to_chemical = self._init_chemical_mapping()
        self.id_to_name = {i: p["name"] for i, p in enumerate(self.pigments)}
        self.name_to_id = {p["name"]: i for i, p in enumerate(self.pigments)}
        
    def _init_pigment_data(self):
        """Pigment data from research paper"""
        return [
            # White pigments (Table 1 in the paper)
            {"name": "kaolin", "color": "white", "formula": "Al2Si2O5(OH)4", 
            "reflectance": [0.95, 0.95, 0.95], "roughness": 0.25,
            "aging": {"yellowing": 0.1, "darkening": 0.1, "fading": 0.2},
            "particle_size": {"mean": 5.0, "std": 1.2, "unit": "micron"}}, 
            
            {"name": "calcium_carbonate", "color": "white", "formula": "CaCO3", 
            "reflectance": [0.92, 0.92, 0.92], "roughness": 0.3,
            "aging": {"yellowing": 0.15, "darkening": 0.12, "fading": 0.1},
            "particle_size": {"mean": 3.5, "std": 0.9, "unit": "micron"}},
            
            {"name": "gypsum", "color": "white", "formula": "CaSO4·2H2O", 
            "reflectance": [0.9, 0.9, 0.9], "roughness": 0.27,
            "aging": {"yellowing": 0.2, "darkening": 0.12, "fading": 0.1},
            "particle_size": {"mean": 8.0, "std": 1.5, "unit": "micron"}},
            
            {"name": "lead_white", "color": "white", "formula": "Pb3(CO3)2(OH)2", 
            "reflectance": [0.93, 0.93, 0.93], "roughness": 0.2,
            "aging": {"yellowing": 0.2, "darkening": 0.6, "fading": 0.05},
            "particle_size": {"mean": 2.5, "std": 0.8, "unit": "micron"}},
            
            # Red pigments (Table 2)
            {"name": "cinnabar", "color": "red", "formula": "HgS", 
            "reflectance": [0.85, 0.1, 0.1], "roughness": 0.4,
            "aging": {"yellowing": 0.05, "darkening": 0.2, "fading": 0.3},
            "particle_size": {"mean": 1.5, "std": 0.5, "unit": "micron"}},
            
            {"name": "vermilion", "color": "red", "formula": "HgS", 
            "reflectance": [0.88, 0.12, 0.1], "roughness": 0.35,
            "aging": {"yellowing": 0.05, "darkening": 0.2, "fading": 0.3},
            "particle_size": {"mean": 1.8, "std": 0.6, "unit": "micron"}},
            
            {"name": "hematite", "color": "red", "formula": "Fe2O3", 
            "reflectance": [0.7, 0.15, 0.15], "roughness": 0.5,
            "aging": {"yellowing": 0.1, "darkening": 0.5, "fading": 0.1},
            "particle_size": {"mean": 4.0, "std": 1.0, "unit": "micron"}},
            
            {"name": "lead_oxide", "color": "red", "formula": "Pb3O4", 
            "reflectance": [0.85, 0.25, 0.1], "roughness": 0.3,
            "aging": {"yellowing": 0.1, "darkening": 0.7, "fading": 0.2},
            "particle_size": {"mean": 3.0, "std": 0.7, "unit": "micron"}},
            
            # Blue pigments (Table 3)
            {"name": "azurite", "color": "blue", "formula": "Cu3(CO3)2(OH)2", 
            "reflectance": [0.1, 0.3, 0.8], "roughness": 0.45,
            "aging": {"yellowing": 0.3, "darkening": 0.4, "fading": 0.2},
            "particle_size": {"mean": 7.0, "std": 1.8, "unit": "micron"}},
            
            {"name": "lapis_lazuli", "color": "blue", "formula": "Na8Ca8(AlSiO4)6S", 
            "reflectance": [0.15, 0.25, 0.85], "roughness": 0.4,
            "aging": {"yellowing": 0.1, "darkening": 0.2, "fading": 0.15},
            "particle_size": {"mean": 10.0, "std": 2.5, "unit": "micron"}},
            
            {"name": "synthetic_blue", "color": "blue", "formula": "Na6Al4Si4S4O20", 
            "reflectance": [0.2, 0.4, 0.82], "roughness": 0.35,
            "aging": {"yellowing": 0.2, "darkening": 0.3, "fading": 0.25},
            "particle_size": {"mean": 5.0, "std": 1.0, "unit": "micron"}},
            
            # Green pigments (Table 4)
            {"name": "malachite", "color": "green", "formula": "Cu2Cl(OH)3", 
            "reflectance": [0.1, 0.75, 0.2], "roughness": 0.4,
            "aging": {"yellowing": 0.25, "darkening": 0.35, "fading": 0.2},
            "particle_size": {"mean": 6.0, "std": 1.2, "unit": "micron"}},
            
            {"name": "atacamite", "color": "green", "formula": "Cu2(CO3)(OH)2", 
            "reflectance": [0.15, 0.7, 0.25], "roughness": 0.45,
            "aging": {"yellowing": 0.3, "darkening": 0.3, "fading": 0.25},
            "particle_size": {"mean": 5.5, "std": 1.0, "unit": "micron"}},
            
            # Yellow pigments (Table 5)
            {"name": "yellow_ochre", "color": "yellow", "formula": "FeO(OH)", 
            "reflectance": [0.85, 0.75, 0.1], "roughness": 0.5,
            "aging": {"yellowing": 0.05, "darkening": 0.4, "fading": 0.2},
            "particle_size": {"mean": 3.5, "std": 0.9, "unit": "micron"}},
            
            {"name": "orpiment", "color": "yellow", "formula": "As2S3", 
            "reflectance": [0.9, 0.8, 0.1], "roughness": 0.35,
            "aging": {"yellowing": 0.1, "darkening": 0.2, "fading": 0.4},
            "particle_size": {"mean": 2.0, "std": 0.6, "unit": "micron"}},
            
            {"name": "realgar", "color": "yellow", "formula": "As4S4", 
            "reflectance": [0.85, 0.6, 0.05], "roughness": 0.4,
            "aging": {"yellowing": 0.1, "darkening": 0.3, "fading": 0.5},
            "particle_size": {"mean": 2.5, "std": 0.7, "unit": "micron"}}
        ]
    
    def _init_color_mapping(self):
        """Map colors to pigment indices based on paper data"""
        color_map = defaultdict(list)
        for i, pigment in enumerate(self.pigments):
            color_map[pigment["color"]].append(i)
        return dict(color_map)
    
    def _init_chemical_mapping(self):
        """Map pigments to chemical compositions for validation"""
        chem_map = {}
        for i, pigment in enumerate(self.pigments):
            chem_map[i] = self._parse_chemical_formula(pigment["formula"])
        return chem_map
    
    def _parse_chemical_formula(self, formula):
        """
        Parse chemical formula to element counts using proper parsing
        """
        # Parse the formula to get element counts
        try:
            elements = parse_formula(formula)
        except Exception as e:
            # Fallback to simplified parsing if error occurs
            print(f"Error parsing formula {formula}: {e}")
            elements = defaultdict(int)
            if 'Cu' in formula:
                elements['Cu'] = formula.count('Cu')
            if 'Fe' in formula:
                elements['Fe'] = formula.count('Fe')
            if 'Hg' in formula:
                elements['Hg'] = formula.count('Hg')
            if 'Pb' in formula:
                elements['Pb'] = formula.count('Pb')
            if 'As' in formula:
                elements['As'] = formula.count('As')
            if 'Ca' in formula:
                elements['Ca'] = formula.count('Ca')
            if 'Al' in formula:
                elements['Al'] = formula.count('Al')
            if 'Si' in formula:
                elements['Si'] = formula.count('Si')
            
        return dict(elements)
    
    def get_reflectance(self, pigment_id):
        """Get reflectance for a pigment by ID"""
        return torch.tensor(self.pigments[pigment_id]["reflectance"])
    
    def get_roughness(self, pigment_id):
        """Get roughness for a pigment by ID"""
        return self.pigments[pigment_id]["roughness"]
    
    def get_aging_factors(self, pigment_id):
        """Get aging factors for a pigment by ID"""
        return self.pigments[pigment_id]["aging"]
    
    def get_pigments_by_color(self, color):
        """Get pigment IDs for a given color category"""
        return self.color_to_pigment.get(color, [])
    
    def get_chemical_signature(self, pigment_id):
        """Get chemical signature for validation"""
        return self.pigment_to_chemical[pigment_id]


# ===== 2. Optical Properties Module =====
class OpticalProperties(nn.Module):
    """
    OpticalProperties module that integrates Dunhuang-specific pigment data from 
    research paper.
    """
    def __init__(self, num_pigments=16, pigment_embedding_dim=32):
        super().__init__()
        self.num_pigments = num_pigments
        self.pigment_embedding_dim = pigment_embedding_dim
        
        # Initialize pigment database
        self.pigment_db = DunhuangPigmentDB()
        
        # Embedding for pigment types
        self.pigment_embedding = nn.Embedding(num_pigments, pigment_embedding_dim)
        
        # Learnable parameters for optical properties
        self.roughness_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Roughness in [0, 1]
        )
        
        self.reflectance_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Sigmoid()  # RGB reflectance in [0, 1]
        )
        
        self.aging_factor_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim + 1, 32),  # +1 for age input
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 9),  # 3 RGB factors for each of (yellowing, darkening, fading)
            nn.Sigmoid()
        )
        
        # Initialize with Dunhuang pigment data
        self._initialize_from_dunhuang_pigments()
        
    def _initialize_from_dunhuang_pigments(self):
        """
        Xavier initialization from database guidance, for better stability.
        """
        nn.init.xavier_normal_(self.pigment_embedding.weight)
        
        with torch.no_grad():
            # Initialize with known reflectance and roughness values
            for i in range(min(self.num_pigments, len(self.pigment_db.pigments))):
                pigment = self.pigment_db.pigments[i]
                
                # Initialize embedding first dimensions with reflectance
                self.pigment_embedding.weight.data[i, 0:3] = torch.tensor(pigment["reflectance"])
                
                # Initialize roughness bias based on database value
                # Find the final layer in roughness prediction
                roughness_final_layer = self.roughness_pred[-2]  # Last linear layer before sigmoid
                if i < roughness_final_layer.bias.shape[0]:
                    roughness_final_layer.bias.data[0] += (pigment["roughness"] - 0.5) * 0.1
                
                # Initialize aging factors in additional dimensions
                if "aging" in pigment:
                    self.pigment_embedding.weight.data[i, 3] = pigment["aging"]["yellowing"]
                    self.pigment_embedding.weight.data[i, 4] = pigment["aging"]["darkening"]
                    self.pigment_embedding.weight.data[i, 5] = pigment["aging"]["fading"]
    
    def _compute_aging_effect(self, pigment_embed, age_factor):
        """
        Compute aging effects with pigment-specific behaviors based on Dunhuang research.
        """
        # Handle input dimensions
        original_shape = pigment_embed.shape
        
        # Ensure pigment_embed is 2D: [batch_size, embedding_dim]
        if len(original_shape) == 3:  # [batch_size, k, embedding_dim]
            batch_size, k, embed_dim = original_shape
            pigment_embed = pigment_embed.reshape(-1, embed_dim)  # [batch_size*k, embedding_dim]
            
            # Ensure age_factor matches pigment_embed
            if len(age_factor.shape) == 1:  # [batch_size]
                age_factor = age_factor.unsqueeze(1).expand(-1, k).reshape(-1)  # [batch_size*k]
            elif len(age_factor.shape) == 2 and age_factor.shape[1] == 1:  # [batch_size, 1]
                age_factor = age_factor.expand(-1, k).reshape(-1)  # [batch_size*k]
        
        # Ensure age_factor is 2D
        if len(age_factor.shape) == 1:
            age_expanded = age_factor.unsqueeze(-1)  # [batch_size, 1] or [batch_size*k, 1]
        else:
            age_expanded = age_factor  # already [batch_size, 1] or [batch_size*k, 1]
        
        # Combine embedding and age
        aging_input = torch.cat([pigment_embed, age_expanded], dim=-1)
        
        # Predict aging factors with pigment-specific behaviors
        aging_factors = self.aging_factor_pred(aging_input)
        
        # Split into different effects
        yellowing = aging_factors[:, 0:3]  # RGB
        darkening = aging_factors[:, 3:6]  # RGB
        fading = aging_factors[:, 6:9]     # RGB
        
        # Reshape if original input was 3D
        if len(original_shape) == 3:
            yellowing = yellowing.reshape(batch_size, k, 3)
            darkening = darkening.reshape(batch_size, k, 3)
            fading = fading.reshape(batch_size, k, 3)
        
        # Create effects dictionary
        return {
            'yellowing': yellowing,
            'darkening': darkening,
            'fading': fading
        }
    
    def _apply_aging(self, reflectance, aging_effects, age_factor):
        """
        Apply aging effects to reflectance based on Dunhuang research.
        Incorporates chemical-specific aging patterns observed in murals.
        """
        # Handle input dimensions
        original_shape = reflectance.shape
        is_3d = len(original_shape) == 3
        
        # Prepare age_factor to match dimensions
        if is_3d:
            if len(age_factor.shape) == 1:  # [batch_size]
                age = age_factor.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            else:  # [batch_size, 1]
                age = age_factor.unsqueeze(2)  # [batch_size, 1, 1]
        else:
            if len(age_factor.shape) == 1:  # [batch_size]
                age = age_factor.unsqueeze(1)  # [batch_size, 1]
            else:  # [batch_size, 1]
                age = age_factor  # already [batch_size, 1]
        
        # Get effects
        yellowing = aging_effects['yellowing']
        darkening = aging_effects['darkening']
        fading = aging_effects['fading']
        
        # Create yellowing mask
        # Lead-based pigments yellow differently than mineral-based ones (from research)
        if is_3d:
            yellowing_mask = torch.tensor([1.0, 0.6, -0.4], device=reflectance.device).view(1, 1, 3)
        else:
            yellowing_mask = torch.tensor([1.0, 0.6, -0.4], device=reflectance.device).view(1, 3)
        
        # Apply yellowing - more pronounced in lead-based white pigments (from research)
        yellowing_effect = 1.0 + yellowing * yellowing_mask * age
        
        # Apply darkening - iron oxide pigments darken more (from research)
        darkening_effect = 1.0 - darkening * age
        
        # Apply fading - arsenic-based pigments (realgar/orpiment) fade more (from research)
        fading_direction = 1.0 - reflectance  # direction toward white
        fading_effect = reflectance + fading * fading_direction * age
        
        # Combine effects - first apply yellowing and darkening
        intermediate = reflectance * yellowing_effect * darkening_effect
        
        # Then blend with fading effect
        if is_3d:
            fade_blend_factor = (age * 0.5)  # [batch_size, 1, 1]
        else:
            fade_blend_factor = (age * 0.5)  # [batch_size, 1]
            
        aged_reflectance = (1 - fade_blend_factor) * intermediate + fade_blend_factor * fading_effect
        aged_reflectance = torch.clamp(aged_reflectance, 0.0, 1.0)
        
        return aged_reflectance
    
    def _apply_particle_size_effects(self, reflectance, roughness, particle_data):
        """
        Apply corrections based on particle size distribution
        - Smaller particles increase saturation but reduce overall reflectance
        - Larger particles increase scattering (roughness) but maintain reflectance
        """
        mean_size = particle_data['mean']
        std_size = particle_data['std']
        
        # Normalize to a reference size of 5 microns
        size_ratio = mean_size / 5.0
        
        # Adjust reflectance based on size
        # Smaller particles appear more saturated but slightly darker
        if size_ratio < 1.0:
            # Enhance saturation by pulling values away from middle gray
            middle = torch.tensor([0.5, 0.5, 0.5], device=reflectance.device)
            reflectance = middle + (reflectance - middle) * (1.0 + (1.0 - size_ratio) * 0.2)
            # Slightly reduce brightness for smaller particles
            reflectance = reflectance * (0.95 + size_ratio * 0.05)
        else:
            # Larger particles maintain reflectance but appear less saturated
            middle = torch.tensor([0.5, 0.5, 0.5], device=reflectance.device)
            reflectance = middle + (reflectance - middle) / (1.0 + (size_ratio - 1.0) * 0.1)
        
        # Adjust roughness based on size
        # Larger particles increase surface roughness
        roughness_mod = roughness * (0.8 + size_ratio * 0.2)
        
        # Apply variability based on standard deviation
        # Higher std means more variation in optical properties
        variability = std_size / mean_size  # Coefficient of variation
        
        # Add subtle texture variation based on particle size variability
        texture_variance = torch.rand_like(roughness) * variability * 0.1
        roughness_final = torch.clamp(roughness_mod + texture_variance, 0.05, 0.95)
        
        return reflectance, roughness_final
        
    def forward(self, pigment_indices, age_factor=None):
        """
        Calculate optical properties for given pigments with aging effects based on 
        Dunhuang research.
        """
        # Handle input dimensions
        original_shape = pigment_indices.shape
        is_2d = len(original_shape) == 2
        
        # Check if input indices are out of range
        pigment_indices = torch.clamp(pigment_indices, 0, self.num_pigments - 1)
        
        # Get pigment embeddings
        if is_2d:
            # If input is [batch_size, k]
            batch_size, k = original_shape
            
            # Flatten for embedding lookup, then restore shape
            flat_indices = pigment_indices.reshape(-1)  # [batch_size*k]
            pigment_embed_flat = self.pigment_embedding(flat_indices)  # [batch_size*k, embed_dim]
            pigment_embed = pigment_embed_flat.reshape(batch_size, k, -1)  # [batch_size, k, embed_dim]
            
            # Predict optical properties for each pigment
            # Flatten for processing
            flat_embed = pigment_embed.reshape(-1, self.pigment_embedding_dim)  # [batch_size*k, embed_dim]
            
            # Calculate reflectance and roughness
            flat_reflectance = self.reflectance_pred(flat_embed)  # [batch_size*k, 3]
            flat_roughness = self.roughness_pred(flat_embed)      # [batch_size*k, 1]
            
            # Restore batch dimensions
            reflectance = flat_reflectance.reshape(batch_size, k, 3)  # [batch_size, k, 3]
            roughness = flat_roughness.reshape(batch_size, k, 1)      # [batch_size, k, 1]
            
        else:
            # If input is [batch_size]
            pigment_embed = self.pigment_embedding(pigment_indices)  # [batch_size, embed_dim]
            
            # Predict basic properties
            reflectance = self.reflectance_pred(pigment_embed)  # [batch_size, 3]
            roughness = self.roughness_pred(pigment_embed)      # [batch_size, 1]
        
        # Apply aging effects if age factor is provided
        if age_factor is not None:
            # Calculate aging effects
            aging_effects = self._compute_aging_effect(pigment_embed, age_factor)
            
            # Apply aging to reflectance
            aged_reflectance = self._apply_aging(reflectance, aging_effects, age_factor)
            
            return {
                'reflectance': aged_reflectance,
                'roughness': roughness,
                'age_factor': age_factor,
                'aging_effects': aging_effects
            }
        
        return {
            'reflectance': reflectance,
            'roughness': roughness
        }


# ===== 3. Color to Pigment Mapper with K-means clustering =====
class ColorToPigmentMapper(nn.Module):
    """
    Maps image color features to probabilistic pigment distributions based on Dunhuang 
    mural research.
    """
    def __init__(self, swin_feat_dim=768, color_feat_dim=128, num_pigments=16):
        super().__init__()
        self.num_pigments = num_pigments
        self.swin_feat_dim = swin_feat_dim
        self.color_feat_dim = color_feat_dim
        
        # Initialize pigment database
        self.pigment_db = DunhuangPigmentDB()
        
        # Color feature extraction - make this more flexible for different input sizes
        self.color_encoder = nn.Sequential(
            nn.Linear(3*256, 256),  # RGB histogram (256 bins per channel)
            nn.ReLU(),
            nn.Linear(256, color_feat_dim)
        )
        
        # Add feature adaptation layer to handle varying input dimensions
        self.feature_adapter = nn.Linear(3, swin_feat_dim)  # Default simple adapter
        
        # Joint feature processing
        self.joint_mlp = nn.Sequential(
            nn.Linear(swin_feat_dim + color_feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_pigments)
        )
        
        # Cultural prior weighting - learned weights for historical accuracy
        self.cultural_prior = nn.Parameter(torch.ones(num_pigments))
        
        # Color name mapping for clustering results
        self.color_centroids = {
            'white': [0.9, 0.9, 0.9],
            'red': [0.8, 0.1, 0.1],
            'blue': [0.1, 0.2, 0.8],
            'green': [0.1, 0.7, 0.2],
            'yellow': [0.9, 0.8, 0.1],
            'black': [0.1, 0.1, 0.1],
            'brown': [0.6, 0.4, 0.2],
            'purple': [0.5, 0.1, 0.5],
            'orange': [0.9, 0.5, 0.1]
        }
        
    def _rgb_to_color_name(self, rgb_values):
        """Map RGB values to the closest named color"""
        min_dist = float('inf')
        closest_color = 'other'
        
        for color_name, centroid in self.color_centroids.items():
            # Calculate Euclidean distance
            dist = np.sqrt(sum([(a - b) ** 2 for a, b in zip(rgb_values, centroid)]))
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
                
        return closest_color
        
    def forward(self, swin_features, color_histogram):
        """
        Predict pigment probability distribution from image features
        """
        # Check input dimensions and adapt if necessary
        if swin_features.size(1) != self.swin_feat_dim:
            # Apply feature adaptation to match expected dimensions
            if swin_features.size(1) == 3:  # Simple RGB features
                swin_features = self.feature_adapter(swin_features)
            else:
                # For other dimensions, use adaptive pooling
                swin_features = F.adaptive_avg_pool1d(
                    swin_features.unsqueeze(2), 
                    self.swin_feat_dim
                ).squeeze(2)
        
        # Ensure color_histogram has correct dimensions
        if color_histogram.size(1) != 3*256:
            # Apply reshaping or padding as necessary
            if len(color_histogram.shape) == 1:
                # If it's a 1D tensor, reshape to batch size 1
                color_histogram = color_histogram.unsqueeze(0)
            
            # If wrong dimension, create a placeholder histogram
            if color_histogram.size(1) != 3*256:
                print(f"Warning: Color histogram has incorrect dimension {color_histogram.size(1)}, creating placeholder")
                color_histogram = torch.zeros((swin_features.size(0), 3*256), device=swin_features.device)
                # Add some random values to avoid all zeros
                color_histogram = color_histogram + torch.rand_like(color_histogram) * 0.1
        
        # Extract color features
        color_feat = self.color_encoder(color_histogram)
        
        # Concatenate features
        joint_feat = torch.cat([swin_features, color_feat], dim=1)
        
        # Predict raw pigment logits
        pigment_logits = self.joint_mlp(joint_feat)
        
        # Apply cultural/historical priors
        weighted_logits = pigment_logits * self.cultural_prior
        
        # Convert to probability distribution
        pigment_probs = F.softmax(weighted_logits, dim=1)
        
        return pigment_probs
    
    def apply_color_constraints(self, pigment_probs, dominant_color):
        """
        Apply constraints based on dominant color and historical usage
        """
        batch_size = pigment_probs.size(0)
        constrained_probs = pigment_probs.clone()
        
        for b in range(batch_size):
            color = dominant_color[b]
            
            # Get valid pigments for this color
            valid_indices = self.pigment_db.get_pigments_by_color(color)
            
            if valid_indices:
                # Create mask of valid pigments
                mask = torch.zeros(self.num_pigments, device=pigment_probs.device)
                mask[valid_indices] = 1.0
                
                # Apply mask and renormalize
                constrained_probs[b] = pigment_probs[b] * mask
                constrained_probs[b] = constrained_probs[b] / (constrained_probs[b].sum() + 1e-8)
        
        return constrained_probs
    
class PigmentValidationMetrics:
    """
    Comprehensive validation metrics for the pigment model, not fully implemented, need to add more
    """
    def __init__(self, historical_data_path=None):
        # Load historical data if available. Not implemented yet, please comment here if you have any ideas...
        self.historical_data = None
        if historical_data_path and os.path.exists(historical_data_path):
            with open(historical_data_path, 'r') as f:
                self.historical_data = json.load(f)
    
    def compute_cultural_accuracy(self, predicted_pigments, region_info):
        """
        Validate pigment predictions against historical records
        
        Args:
            predicted_pigments: Dict with pigment ids and probabilities
            region_info: Information about the region (cave number, date, etc.)
            
        Returns:
            Cultural accuracy score (0-1)
        """
        if self.historical_data is None or not region_info:
            return None
            
        # Look up historical pigments for this region and period
        period = region_info.get('period', 'unknown')
        cave = region_info.get('cave', 'unknown')
        
        # Find matching historical record
        historical_pigments = None
        for record in self.historical_data:
            if record['period'] == period and record['cave'] == cave:
                historical_pigments = record['pigments']
                break
        
        if not historical_pigments:
            return None
        
        # Calculate accuracy based on historical pigment presence
        correct = 0
        total = 0
        
        # Get top predicted pigments
        if isinstance(predicted_pigments, torch.Tensor):
            pred_pigments = predicted_pigments.detach().cpu().numpy()
        else:
            pred_pigments = predicted_pigments
            
        top_pigments = np.argsort(pred_pigments)[-5:]  # Top 5 pigments
        
        # Check against historical records
        for pigment_id in range(len(pred_pigments)):
            is_predicted = pigment_id in top_pigments
            is_historical = pigment_id in historical_pigments
            
            if is_predicted == is_historical:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_spectral_similarity(self, pred_reflectance, gt_reflectance=None):
        """
        Compute similarity between predicted and ground truth reflectance
        Even with RGB values, we can compute a basic similarity metric
        
        Args:
            pred_reflectance: Predicted RGB reflectance
            gt_reflectance: Ground truth RGB reflectance (if available, I think we can just use the original image? Not sure...)
            
        Returns:
            Similarity score (0-1)
        """
        if gt_reflectance is None:
            return None
            
        # Convert to numpy if needed
        if isinstance(pred_reflectance, torch.Tensor):
            pred_rgb = pred_reflectance.detach().cpu().numpy()
        else:
            pred_rgb = pred_reflectance
            
        if isinstance(gt_reflectance, torch.Tensor):
            gt_rgb = gt_reflectance.detach().cpu().numpy()
        else:
            gt_rgb = gt_reflectance
        
        # Compute Euclidean distance in RGB space
        rgb_distance = np.sqrt(np.sum((pred_rgb - gt_rgb) ** 2))
        
        # Convert to similarity score (0-1)
        similarity = np.exp(-rgb_distance * 2)  # Exponential falloff
        
        return similarity
    
    def compute_aging_accuracy(self, pred_aging, gt_aging=None):
        """
        Compute accuracy of aging predictions
        
        Args:
            pred_aging: Predicted aging effects
            gt_aging: Ground truth aging data (if available. This is the most confusing part, I think we should research more on chemstry papers)
            
        Returns:
            Aging accuracy score (0-1)
        """
        if gt_aging is None:
            return None
            
        # Extract aging components
        if isinstance(pred_aging, dict):
            pred_yellowing = pred_aging.get('yellowing', 0)
            pred_darkening = pred_aging.get('darkening', 0)
            pred_fading = pred_aging.get('fading', 0)
        else:
            # Default extraction if not in dict format
            pred_yellowing = pred_darkening = pred_fading = pred_aging
        
        # Same for ground truth
        if isinstance(gt_aging, dict):
            gt_yellowing = gt_aging.get('yellowing', 0)
            gt_darkening = gt_aging.get('darkening', 0)
            gt_fading = gt_aging.get('fading', 0)
        else:
            gt_yellowing = gt_darkening = gt_fading = gt_aging
        
        # Convert to numpy if needed
        if isinstance(pred_yellowing, torch.Tensor):
            pred_yellowing = pred_yellowing.detach().cpu().numpy()
        if isinstance(pred_darkening, torch.Tensor):
            pred_darkening = pred_darkening.detach().cpu().numpy()
        if isinstance(pred_fading, torch.Tensor):
            pred_fading = pred_fading.detach().cpu().numpy()
        
        # Compute mean absolute error for each component
        yellowing_error = np.mean(np.abs(pred_yellowing - gt_yellowing))
        darkening_error = np.mean(np.abs(pred_darkening - gt_darkening))
        fading_error = np.mean(np.abs(pred_fading - gt_fading))
        
        # Compute overall error
        total_error = (yellowing_error + darkening_error + fading_error) / 3
        
        # Convert to accuracy score (0-1)
        accuracy = np.exp(-total_error * 5)  # Exponential falloff
        
        return accuracy
    
    def compute_all_metrics(self, predictions, ground_truth=None, region_info=None):
        """
        Compute all available validation metrics
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth data (if available)
            region_info: Information about the region (if available)
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {}
        
        # Extract prediction components
        pigment_probs = predictions.get('pigment_probs', None)
        reflectance = predictions.get('reflectance', None)
        aging_effects = predictions.get('aging_effects', None)
        
        # Extract ground truth components if available
        gt_pigments = None
        gt_reflectance = None
        gt_aging = None
        
        if ground_truth is not None:
            gt_pigments = ground_truth.get('pigment_probs', None)
            gt_reflectance = ground_truth.get('reflectance', None)
            gt_aging = ground_truth.get('aging_effects', None)
        
        # Compute cultural accuracy
        if pigment_probs is not None and region_info is not None:
            metrics['cultural_accuracy'] = self.compute_cultural_accuracy(
                pigment_probs, region_info
            )
        
        # Compute spectral similarity
        if reflectance is not None:
            metrics['spectral_similarity'] = self.compute_spectral_similarity(
                reflectance, gt_reflectance
            )
        
        # Compute aging accuracy
        if aging_effects is not None:
            metrics['aging_accuracy'] = self.compute_aging_accuracy(
                aging_effects, gt_aging
            )
        
        return metrics

class ThermodynamicValidator:    
    """
    Validates pigment mixtures for chemical stability based on materials science principles/
    We need to continue updating this as we find more research papers.
    """
    def __init__(self, pigment_db):
        self.pigment_db = pigment_db
        
        # Define incompatible pigment pairs (based on chemical reactivity)
        # Format: (pigment1_id, pigment2_id, reason)
        self.incompatible_pairs = [
            # Lead white and sulfide-containing pigments
            (3, 4, "Lead white reacts with sulfides in cinnabar"),
            (3, 5, "Lead white reacts with sulfides in vermilion"),
            
            # Copper-based pigments and sulfides
            (8, 4, "Copper-based azurite deteriorates with sulfide-containing pigments"),
            (8, 5, "Copper-based azurite deteriorates with sulfide-containing pigments"),
            (11, 4, "Copper-based malachite deteriorates with sulfide-containing pigments"),
            (11, 5, "Copper-based malachite deteriorates with sulfide-containing pigments"),
            
            # Arsenic and copper-based pigments
            (14, 8, "Orpiment can react with copper-based pigments"),
            (14, 11, "Orpiment can react with copper-based pigments"),
            (15, 8, "Realgar can react with copper-based pigments"),
            (15, 11, "Realgar can react with copper-based pigments"),
        ]
        
        # Define pH sensitivity for pigments
        # Format: pigment_id: {"acid": sensitivity, "alkaline": sensitivity}
        # where sensitivity is 0 (stable) to 1 (highly sensitive)
        self.ph_sensitivity = {
            3: {"acid": 0.8, "alkaline": 0.1},      # Lead white deteriorates in acidic environment
            4: {"acid": 0.1, "alkaline": 0.1},      # Cinnabar is fairly stable
            8: {"acid": 0.9, "alkaline": 0.1},      # Azurite deteriorates in acidic environment
            11: {"acid": 0.7, "alkaline": 0.2},     # Malachite sensitive to acids
            14: {"acid": 0.2, "alkaline": 0.6},     # Orpiment more sensitive to alkaline
        }
    
    def check_mixture_stability(self, pigment_indices, mixing_ratios):
        """
        Check if a mixture of pigments is chemically stable
        
        Args:
            pigment_indices: Tensor of pigment indices
            mixing_ratios: Tensor of mixing ratios
            
        Returns:
            Dictionary with stability information and warnings
        """
        # Convert to numpy if necessary
        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().numpy()
        else:
            indices = pigment_indices
            
        if isinstance(mixing_ratios, torch.Tensor):
            ratios = mixing_ratios.detach().cpu().numpy()
        else:
            ratios = mixing_ratios
        
        # Initialize results
        stability_score = 1.0  # Start with perfect stability
        warnings = []
        unstable_pairs = []
        
        # Check for incompatible pigment pairs
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                pigment1 = indices[i]
                pigment2 = indices[j]
                
                # Skip if either pigment has negligible concentration
                if ratios[i] < 0.05 or ratios[j] < 0.05:
                    continue
                
                # Check if this pair is in the incompatible list
                for p1, p2, reason in self.incompatible_pairs:
                    if (pigment1 == p1 and pigment2 == p2) or (pigment1 == p2 and pigment2 == p1):
                        # Calculate instability based on mixing ratios
                        instability = min(ratios[i], ratios[j]) * 4.0
                        stability_score -= instability * 0.4  # Reduce stability score
                        stability_score = max(0.0, min(1.0, stability_score))
                        
                        # Add warning
                        p1_name = self.pigment_db.id_to_name.get(pigment1, f"Pigment {pigment1}")
                        p2_name = self.pigment_db.id_to_name.get(pigment2, f"Pigment {pigment2}")
                        warnings.append(f"Incompatible pigments: {p1_name} and {p2_name}. {reason}")
                        unstable_pairs.append((i, j))
        
        # Ensure stability score is in valid range
        stability_score = max(0.0, min(1.0, stability_score))
        
        # Return results
        return {
            "stability_score": stability_score,
            "warnings": warnings,
            "unstable_pairs": unstable_pairs,
            "is_stable": stability_score > 0.7  # Consider stable if score > 0.7, some research paper says 0.67, we just go with 0.7
        }
    
    def suggest_corrections(self, pigment_indices, mixing_ratios):
        """
        Suggest corrections to improve stability of a pigment mixture
        
        Args:
            pigment_indices: Tensor of pigment indices
            mixing_ratios: Tensor of mixing ratios
            
        Returns:
            Corrected mixing ratios and suggested alternatives
        """
        # First check stability
        stability_info = self.check_mixture_stability(pigment_indices, mixing_ratios)
        
        # If already stable, return as is
        if stability_info["is_stable"]:
            return {
                "corrected_ratios": mixing_ratios,
                "suggested_alternatives": [],
                "stability_info": stability_info
            }
        
        # Convert to numpy if necessary
        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().numpy()
        else:
            indices = pigment_indices
            
        if isinstance(mixing_ratios, torch.Tensor):
            ratios = mixing_ratios.detach().cpu().numpy().copy()  # Make a copy
        else:
            ratios = mixing_ratios.copy()  # Make a copy
        
        # Get unstable pairs
        unstable_pairs = stability_info["unstable_pairs"]
        
        # Adjust ratios to minimize interaction
        for i, j in unstable_pairs:
            # Find the pigment with lower concentration
            if ratios[i] <= ratios[j]:
                minor_idx = i
                major_idx = j
            else:
                minor_idx = j
                major_idx = i
            
            # Reduce the minor pigment's concentration by 50%
            ratios[minor_idx] *= 0.5
            
            # Suggest alternative pigments
            pigment_to_replace = indices[minor_idx]
            alternatives = self._suggest_alternative_pigments(pigment_to_replace, indices)
        
        # Renormalize ratios
        ratios = ratios / np.sum(ratios)
        
        return {
            "corrected_ratios": ratios,
            "suggested_alternatives": alternatives,
            "stability_info": self.check_mixture_stability(pigment_indices, ratios)
        }
    
    def _suggest_alternative_pigments(self, pigment_id, current_mixture):
        """
        Suggest alternative pigments that are more compatible
        
        Args:
            pigment_id: ID of the pigment to replace
            current_mixture: Current mixture of pigments
            
        Returns:
            List of suggested alternative pigments
        """
        # Get color group of the pigment
        pigment_color = None
        for p in self.pigment_db.pigments:
            if self.pigment_db.name_to_id.get(p["name"]) == pigment_id:
                pigment_color = p["color"]
                break
        
        if not pigment_color:
            return []
        
        # Get all pigments of the same color
        same_color_pigments = self.pigment_db.get_pigments_by_color(pigment_color)
        
        # Filter out pigments already in the mixture and the one to replace
        alternatives = [p for p in same_color_pigments if p not in current_mixture and p != pigment_id]
        
        # Check stability of each alternative with the current mixture
        stable_alternatives = []
        for alt in alternatives:
            # Create a test mixture
            test_mixture = [p for p in current_mixture if p != pigment_id] + [alt]
            test_ratios = np.ones(len(test_mixture)) / len(test_mixture)
            
            # Check stability
            stability = self.check_mixture_stability(test_mixture, test_ratios)
            if stability["is_stable"]:
                # Get name of alternative
                alt_name = "Unknown"
                for p in self.pigment_db.pigments:
                    if self.pigment_db.name_to_id.get(p["name"]) == alt:
                        alt_name = p["name"]
                        break
                
                stable_alternatives.append({
                    "id": alt,
                    "name": alt_name,
                    "stability_score": stability["stability_score"]
                })
        
        # Sort by stability score
        stable_alternatives.sort(key=lambda x: x["stability_score"], reverse=True)
        
        return stable_alternatives

# ===== 4. Pigment Constraint Model with Mixing Stability =====
class PigmentModel(nn.Module):
    """
    Predicts optimal pigment mixing and application for image restoration based on Dunhuang 
    mural research.
    """
    def __init__(
        self, 
        num_pigments=16,
        feature_dim=768,
        hidden_dim=256,
        k=3,
        in_chans=3
    ):
        super().__init__()
        self.num_pigments = num_pigments
        self.feature_dim = feature_dim
        self.k = k
        self.in_chans = in_chans
        self.pigment_db = DunhuangPigmentDB()
        
        # Color to pigment mapping
        self.color_mapper = ColorToPigmentMapper(feature_dim, 128, num_pigments)
        
        # Optical properties with Dunhuang data
        self.optical_properties = OpticalProperties(num_pigments)
        
        # Feature to mixing ratio mapping
        self.ratio_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pigments),
            nn.Softmax(dim=-1)
        )
        
        # Age predictor (estimates pigment age from image features)
        self.age_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Map from optical properties back to RGB with separate inputs for clarity
        self.rgb_mapper = nn.Sequential(
            nn.Linear(3 + 1 + self.k, 32),  # reflectance(3) + roughness(1) + ratios(k)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

        # Initialize validators
        self.thermodynamic_validator = ThermodynamicValidator(self.pigment_db)
        self.validation_metrics = PigmentValidationMetrics()   # Note this is not implemented yet!
        
        # Color histogram generator
        self.color_histogram_generator = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(16),  # 16x16 spatial resolution
            nn.Flatten(),
            nn.Linear(64*16*16, 3*256)  # 256 bins per RGB channel
        )
        
        # Layer decomposition module
        self.layer_decomposition = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_chans, kernel_size=1)
        )
        
        # Physical pigment rendering module
        self.physical_renderer = nn.Sequential(
            nn.Conv2d(in_chans + 5, 32, kernel_size=3, padding=1),  # + 5 for pigment properties
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_chans, kernel_size=1)
        )
        
        # Final adjustment module
        self.final_adjustment = nn.Sequential(
            nn.Conv2d(in_chans * 2, 32, kernel_size=3, padding=1),  # Initial + Refined
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_chans, kernel_size=1)
        )
    
    def detect_dominant_colors(self, x):
        """Detect dominant colors using K-means clustering for pigment model"""
        batch_size = x.shape[0]
        colors = []
        
        # Process each image in the batch
        for b in range(batch_size):
            img = x[b].detach().cpu().numpy()  # [3, H, W]
            
            # Reshape to pixels
            pixels = img.transpose(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
            
            # Sample pixels for efficiency
            sample_size = min(1000, pixels.shape[0])
            indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
            pixel_sample = pixels[indices]
            
            # Use K-means clustering to find dominant colors
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(pixel_sample)
            
            # Find the most common cluster
            cluster_labels = kmeans.predict(pixel_sample)
            most_common_cluster = np.argmax(np.bincount(cluster_labels))
            
            # Get the centroid of the most common cluster
            dominant_color = kmeans.cluster_centers_[most_common_cluster]
            
            # Map to color name using pigment database
            color_name = self._rgb_to_color_name(dominant_color)
            colors.append(color_name)
        
        return colors

    def _rgb_to_color_name(self, rgb_values):
        """Map RGB values to color names defined in the Dunhuang pigment system"""
        # Use the color mapping from the pigment database, we can add more centers if have more research papers
        color_map = {
            'white': [0.9, 0.9, 0.9],
            'red': [0.8, 0.2, 0.2],
            'blue': [0.2, 0.2, 0.8],
            'green': [0.2, 0.7, 0.2],
            'yellow': [0.9, 0.8, 0.2]
        }
        
        min_dist = float('inf')
        closest_color = 'other'
        
        for color_name, color_rgb in color_map.items():
            # Calculate Euclidean distance
            dist = np.sqrt(np.sum((np.array(rgb_values) - np.array(color_rgb)) ** 2))
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
        
        return closest_color
    
    def _extract_pigment_features(self, image):
        """Extract features for pigment analysis"""
        # We just use simple feature extraction, we can replace it with more advanced methods, but this should be good enough
        pooled_features = F.adaptive_avg_pool2d(self.layer_decomposition(image), 1).squeeze(-1).squeeze(-1)

        if hasattr(self, 'feature_extractor'): # Future work
            return self.feature_extractor(image)
        return pooled_features
    
    def _compute_color_similarity(self, image, reference_color):
        """Compute color similarity between image and reference color"""
        # Calculate color difference using Gaussian RBF kernel
        diff = torch.sum((image - reference_color)**2, dim=1, keepdim=True)
        sigma = 0.1     # Adjustable parameter for softness
        similarity = torch.exp(-diff / (2 * sigma**2))
        
        return similarity
    
    def _decompose_to_pigment_layers(self, image, pigment_info):
        """Decompose image into separate pigment layers"""
        batch_size = image.shape[0]
        k = pigment_info['top_pigments'].shape[1]  # Number of pigments
        layers = []
        
        # Get color centroids for each pigment
        for i in range(k):
            pigment_indices = pigment_info['top_pigments'][:, i]
            
            # Get reference colors for each pigment
            ref_colors = []
            for b in range(batch_size):
                pigment_id = pigment_indices[b].item()
                if pigment_id < len(self.pigment_db.pigments):
                    ref_color = torch.tensor(self.pigment_db.pigments[pigment_id]["reflectance"])
                    ref_colors.append(ref_color)
                else:
                    # Fallback if the given image has colors not in the database, should be rare
                    ref_colors.append(torch.tensor([0.5, 0.5, 0.5]))
            
            ref_colors = torch.stack(ref_colors).to(image.device).view(batch_size, 3, 1, 1)
            
            # Create color similarity mask
            similarity = self._compute_color_similarity(image, ref_colors)
            
            # Extract layer using soft segmentation
            layer = image * similarity
            layers.append(layer)
        
        return layers
    
    def _apply_pigment_physics(self, layer, pigment_id, age_factor, roughness, batch_idx=0):
        """Apply physical properties to pigment layer"""
        # Get pigment data
        if pigment_id < len(self.pigment_db.pigments):
            pigment = self.pigment_db.pigments[pigment_id]
        else:
            # Fallback to default pigment properties
            pigment = {
                "reflectance": [0.5, 0.5, 0.5],
                "roughness": 0.5,
                "aging": {"yellowing": 0.2, "darkening": 0.2, "fading": 0.2},
                "particle_size": {"mean": 5.0, "std": 1.0}
            }
        
        # Create property tensor for physical rendering
        # [reflectance(3), roughness(1), age(1)]
        age = age_factor[batch_idx].item()
        props = torch.cat([
            torch.tensor(pigment["reflectance"]).to(layer.device).view(1, 3, 1, 1),
            torch.tensor([pigment["roughness"]]).to(layer.device).view(1, 1, 1, 1),
            torch.tensor([age]).to(layer.device).view(1, 1, 1, 1)
        ], dim=1)
        
        # Concatenate properties with layer for conditioning
        conditioned = torch.cat([layer, props.expand(-1, -1, layer.shape[2], layer.shape[3])], dim=1)
        
        # Apply physical rendering
        rendered = self.physical_renderer(conditioned)
        
        # Apply aging effects
        yellowing = pigment["aging"]["yellowing"] * age
        darkening = pigment["aging"]["darkening"] * age
        fading = pigment["aging"]["fading"] * age
        
        # Create aging mask
        aging_mask = torch.tensor([
            1.0 + yellowing - darkening,        # Red channel
            1.0 + yellowing * 0.6 - darkening,  # Green channel
            1.0 - yellowing * 0.4 - darkening   # Blue channel
        ]).to(layer.device).view(1, 3, 1, 1)
        
        # Apply aging
        aged = rendered * aging_mask
        
        # Apply fading (shift toward white)
        faded = aged + fading * (1.0 - aged)
        
        return torch.clamp(faded, 0, 1)
    
    def _physically_mix_layers(self, layers, ratios):
        """Mix pigment layers using physical pigment mixing rules"""
        batch_size = layers[0].shape[0]
        
        # Start with black canvas
        mixed = torch.zeros_like(layers[0])
        
        # Apply simplified Kubelka-Munk theory (We can apply more accurate Kubelka-Munk theory later if have time)
        for i, layer in enumerate(layers):
            ratio = ratios[:, i].view(batch_size, 1, 1, 1)
            
            # Absorption coefficient (simplified)
            K = 1.0 - layer
            
            # Scattering coefficient (simplified)
            S = layer
            
            # Apply mixing with concentration-dependent weights
            mixed = mixed + (layer * ratio) - (mixed * layer * ratio * 0.5)
        
        return torch.clamp(mixed, 0, 1)
    
    def _apply_particle_size_effects(self, reflectance, roughness, particle_data):
        """
        Just for testing purpose.
        """
        return self.optical_properties._apply_particle_size_effects(
            reflectance, roughness, particle_data
        )
    
    def _final_adjustment(self, refined_image, initial_image):
        """Apply final adjustment to ensure natural appearance"""
        # Combine initial and refined images
        combined = torch.cat([initial_image, refined_image], dim=1)
        
        # Apply final adjustment network
        adjusted = self.final_adjustment(combined)
        
        return torch.clamp(adjusted, 0, 1)

    def _adapt_features(self, features):
        """Adapt features to the expected dimension"""
        # Check if features need adaptation
        if features.size(1) != self.feature_dim:
            print(f"Adapting features from shape {features.shape} to feature_dim {self.feature_dim}")
            
            # If simple RGB features (3 dimensions)
            if features.size(1) == 3:
                if not hasattr(self, 'feature_adapter'):
                    self.feature_adapter = nn.Linear(3, self.feature_dim).to(features.device)
                adapted_features = self.feature_adapter(features)
            else:
                # For other dimensions
                adapted_features = F.adaptive_avg_pool1d(
                    features.unsqueeze(2), 
                    self.feature_dim
                ).squeeze(2)
            
            return adapted_features
        
        # Return original features if already correct dimension
        return features
 
    def forward(self, features, color_histogram, dominant_color=None):
        """
        Predict pigment properties from image features with historical accuracy for Dunhuang murals.
        """
        batch_size = features.shape[0]
        features = self._adapt_features(features)
        
        # Predict pigment probabilities
        pigment_probs = self.color_mapper(features, color_histogram)
        
        # Apply color constraints if dominant color is provided
        if dominant_color is not None:
            pigment_probs = self.color_mapper.apply_color_constraints(pigment_probs, dominant_color)
        
        # Predict mixing ratios
        mixing_ratios = self.ratio_predictor(features)
        
        # Predict age factor
        age_factor = self.age_predictor(features).squeeze(-1)  # [batch_size]
        
        # Combine probabilities and mixing ratios for final pigment distribution
        # Add small epsilon to avoid zero values (improved stability)
        final_pigment_dist = (pigment_probs * mixing_ratios) + 1e-6
        final_pigment_dist = final_pigment_dist / final_pigment_dist.sum(dim=1, keepdim=True)
        
        # Get top-k pigments
        top_probs, top_indices = torch.topk(final_pigment_dist, self.k, dim=1)
        
        # Normalize top-k probabilities
        normalized_probs = top_probs / top_probs.sum(dim=1, keepdim=True)
        
        # Get optical properties for top pigments
        props = self.optical_properties(top_indices, age_factor)
        
        # Extract reflectance and roughness
        reflectance = props['reflectance']  # [batch_size, k, 3]
        roughness = props['roughness']      # [batch_size, k, 1]

        # Apply particle size effects
        batch_size = top_indices.size(0)
        for b in range(batch_size):
            for i in range(self.k):
                pigment_id = top_indices[b, i].item()
                if pigment_id < len(self.pigment_db.pigments):
                    pigment = self.pigment_db.pigments[pigment_id]
                    if 'particle_size' in pigment:
                        # Apply particle size effects to individual pigment
                        ref, rough = self._apply_particle_size_effects(
                            reflectance[b, i], 
                            roughness[b, i],
                            pigment['particle_size']
                        )
                        reflectance[b, i] = ref
                        roughness[b, i] = rough

        # Validate stability of pigment mixtures
        stability_info = []
        for b in range(batch_size):
            # Check thermodynamic stability
            stability = self.thermodynamic_validator.check_mixture_stability(
                top_indices[b], normalized_probs[b]
            )
            
            # If unstable, get corrections
            if not stability["is_stable"]:
                corrections = self.thermodynamic_validator.suggest_corrections(
                    top_indices[b], normalized_probs[b]
                )
                
                # Apply corrections if available
                if "corrected_ratios" in corrections and len(corrections["corrected_ratios"]) == len(normalized_probs[b]):
                    corrected_ratios = torch.tensor(
                        corrections["corrected_ratios"], 
                        device=normalized_probs.device, 
                        dtype=normalized_probs.dtype
                    )
                    normalized_probs[b] = corrected_ratios
            
            stability_info.append(stability)
        
        # Calculate weighted reflectance (mixed pigments)
        weighted_reflectance = reflectance * normalized_probs.unsqueeze(-1)
        mixed_reflectance = torch.sum(weighted_reflectance, dim=1)  # [batch_size, 3]
        
        # Calculate weighted roughness
        weighted_roughness = roughness * normalized_probs.unsqueeze(-1)
        mixed_roughness = torch.sum(weighted_roughness, dim=1)  # [batch_size, 1]
        
        # Combine properties for RGB mapping
        rgb_input = torch.cat([
            mixed_reflectance,      # [batch_size, 3]
            mixed_roughness,        # [batch_size, 1]
            normalized_probs        # [batch_size, k]
        ], dim=-1)                  # [batch_size, 3+1+k]
        
        # Map to RGB values
        rgb_values = self.rgb_mapper(rgb_input)  # [batch_size, 3]
        
        return {
            'pigment_probs': final_pigment_dist,
            'top_pigments': top_indices,
            'selected_ratios': normalized_probs,
            'reflectance': mixed_reflectance,
            'roughness': mixed_roughness,
            'rgb_values': rgb_values,
            'age_factor': age_factor,
            'stability_info': stability_info,
            'validation_metrics': self.validation_metrics.compute_all_metrics(
                {'pigment_probs': final_pigment_dist, 'reflectance': mixed_reflectance, 'aging_effects': props.get('aging_effects')},
                None  # No ground truth available during inference
            )
        }
    
    def process_image(self):
        """Process the loaded image with manual analysis, this is just for temorary testing"""
        if not hasattr(self, 'image_tensor'):
            raise ValueError("No image loaded. Call load_image() first.")
        
        print("Analyzing image colors...")
        
        # Convert image to numpy for analysis
        img_np = self.image_tensor.permute(1, 2, 0).numpy()
        
        # Extract dominant colors using K-means clustering
        pixels = img_np.reshape(-1, 3)
        sample_size = min(5000, pixels.shape[0])
        indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
        pixel_sample = pixels[indices]
        
        # Find dominant colors
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixel_sample)
        
        # Get cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_
        
        # Map colors to pigments
        top_pigments = []
        selected_ratios = []
        
        for color in dominant_colors:
            best_match = None
            best_score = float('inf')
            
            # Find best matching pigment
            for i, pigment in enumerate(self.pigment_db.pigments):
                pigment_color = np.array(pigment["reflectance"])
                distance = np.sum((color - pigment_color) ** 2)
                
                if distance < best_score:
                    best_score = distance
                    best_match = i
            
            top_pigments.append(best_match)
            selected_ratios.append(1.0 / len(dominant_colors))  # Equal ratios
        
        # Convert to tensors
        top_pigments_tensor = torch.tensor(top_pigments)
        selected_ratios_tensor = torch.tensor(selected_ratios)
        
        # Store results
        self.results = {
            'restored_image': self.image_tensor.unsqueeze(0),  # Just use original image
            'initial_image': self.image_tensor.unsqueeze(0),
            'pigment_info': {
                'top_pigments': top_pigments_tensor.unsqueeze(0),
                'selected_ratios': selected_ratios_tensor.unsqueeze(0),
                'reflectance': torch.tensor(np.mean(dominant_colors, axis=0)).unsqueeze(0),
                'roughness': torch.tensor([0.4]).unsqueeze(0),  # Default value
                'age_factor': torch.tensor([0.5]).unsqueeze(0)  # Default value
            },
            'pigment_layers': [self.image_tensor.unsqueeze(0) * r for r in selected_ratios]
        }
        
        print("Image analysis complete")
        return self.results