import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from scipy.signal import argrelextrema
from scipy import interpolate
import time
import warnings
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.ndimage import sobel

from pigment_database import DunhuangPigmentDB

# ===== Color to Pigment Mapper =====
class ColorToPigmentMapper(nn.Module):
    """
    Mapper for converting image colors to historically accurate pigment distributions.
    """
    def __init__(self, swin_feat_dim=768, color_feat_dim=128, num_pigments=35, 
                min_clusters=8, max_clusters=20, cluster_threshold=0.01,
                spectral_bands=31, use_dbscan=False, use_ml_mapping=True, 
                device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_pigments = num_pigments
        self.swin_feat_dim = swin_feat_dim
        self.color_feat_dim = color_feat_dim
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.cluster_threshold = cluster_threshold
        self.spectral_bands = spectral_bands
        self.use_dbscan = use_dbscan
        self.use_ml_mapping = use_ml_mapping
        self.wavelengths = torch.linspace(400, 700, spectral_bands, device=self.device)
        self.pigment_db = DunhuangPigmentDB(spectral_resolution=spectral_bands)
        self.cie_cmf = self._init_cie_color_matching().to(self.device)
        self._init_pigment_name_map()
        self.microstructure_params = self._init_microstructure_params()
        self.spectral_profiles = self._init_spectral_profiles()
        self._update_pigment_name_map()

        self.color_encoder = nn.Sequential(
            nn.Linear(3*256, 512),  
            nn.LayerNorm(512),    
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, color_feat_dim)
        )
        
        self.spectral_encoder = nn.Sequential(
            nn.Linear(spectral_bands, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, color_feat_dim // 2)
        )

        self.feature_adapter = nn.Linear(3, swin_feat_dim)
        color_and_spectral_dim = color_feat_dim + color_feat_dim // 2
        combined_feature_dim = swin_feat_dim + color_and_spectral_dim
        self.joint_mlp = nn.Sequential(
            nn.Linear(combined_feature_dim, 512), 
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, num_pigments)
        )

        color_and_spectral_dim = color_feat_dim + (color_feat_dim // 2)
        self.pigment_embedding = nn.Embedding(num_pigments, color_and_spectral_dim)
        self._init_pigment_embeddings()
        self.pigment_prior = nn.Parameter(torch.ones(num_pigments))
        self.style_modulation = nn.Parameter(torch.ones(num_pigments))
        
        # BRDF parameter estimation network
        self.brdf_estimator = nn.Sequential(
            nn.Linear(color_and_spectral_dim + num_pigments, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 5),  # roughness, specular, metallic, anisotropy, subsurface
            nn.Sigmoid() 
        )
        
        # Fluorescence parameter estimation
        self.fluorescence_estimator = nn.Sequential(
            nn.Linear(color_and_spectral_dim + num_pigments, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 3),  # emission strength, peak wavelength shift, bandwidth
            nn.Sigmoid() 
        )

        self.mixture_handler = nn.Sequential(
            nn.Linear(num_pigments * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_pigments),
            nn.Softmax(dim=-1)
        )

        self.device_initialized = True
        self.to(self.device)
    
    def _init_cie_color_matching(self):
        """Initialize spectral to RGB conversion using CIE standard observer data"""
        # Use the 1931 2° Standard Observer data
        wavelengths_nm = np.linspace(400, 700, self.spectral_bands)
        
        # Standard CIE 1931 2° Standard Observer values 
        cie_wavelengths = np.arange(400, 710, 10)
        
        # CIE 1931 2° Standard Observer values (x̄)
        x_bar_values = [0.0143, 0.0435, 0.1344, 0.2839, 0.3483, 0.3362, 0.2908, 0.1954, 0.0956, 0.0320, 
                    0.0049, 0.0093, 0.0633, 0.1655, 0.2904, 0.4334, 0.5945, 0.7621, 0.9163, 1.0000, 
                    0.9548, 0.8689, 0.7774, 0.6583, 0.5280, 0.3981, 0.2835, 0.1798, 0.1076, 0.0601, 0.0320]
        
        # CIE 1931 2° Standard Observer values (ȳ)
        y_bar_values = [0.0004, 0.0012, 0.0040, 0.0116, 0.0230, 0.0380, 0.0600, 0.0910, 0.1390, 0.2080, 
                    0.3230, 0.5030, 0.7100, 0.8620, 0.9540, 0.9950, 0.9950, 0.9520, 0.8700, 0.7570, 
                    0.6310, 0.5030, 0.3810, 0.2650, 0.1750, 0.1070, 0.0610, 0.0320, 0.0170, 0.0082, 0.0041]
        
        # CIE 1931 2° Standard Observer values (z̄)
        z_bar_values = [0.0679, 0.2074, 0.6456, 1.3856, 1.7471, 1.7721, 1.6692, 1.2876, 0.8130, 0.4652, 
                    0.2720, 0.1582, 0.0782, 0.0422, 0.0203, 0.0087, 0.0039, 0.0021, 0.0017, 0.0011, 
                    0.0008, 0.0003, 0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]
        
        x_function = np.interp(wavelengths_nm, cie_wavelengths, x_bar_values)
        y_function = np.interp(wavelengths_nm, cie_wavelengths, y_bar_values)
        z_function = np.interp(wavelengths_nm, cie_wavelengths, z_bar_values)
        
        # XYZ to RGB conversion matrix
        xyz_to_rgb = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        cmf_xyz = np.stack([x_function, y_function, z_function], axis=0)
        cmf_rgb = np.dot(xyz_to_rgb, cmf_xyz)
        if np.max(cmf_rgb) > 0:
            cmf_rgb = cmf_rgb / np.max(cmf_rgb)
        
        return torch.tensor(cmf_rgb, dtype=torch.float32)
    
    def _init_spectral_profiles(self):
        """Initialize spectral profiles with full hyperspectral data"""
        profiles = {}
        for color, data in {
            'white': {'peak': 0.92, 'peak_wavelength': None, 'variance': 0.02,
                     'possible_pigments': ['kaolin', 'calcium_carbonate', 'gypsum', 'lead_white',
                                          'quartz', 'lead_chloride', 'muscovite', 'talc', 
                                          'anhydrite', 'lead_sulfate'],
                     'excluded_pigments': []},
            'red': {'peak': 0.85, 'peak_wavelength': 650, 'variance': 0.12,
                   'possible_pigments': ['cinnabar', 'vermilion', 'hematite', 'lead_oxide', 
                                        'realgar', 'cinnabar_variant', 'red_ochre', 'lac'],
                   'excluded_pigments': ['gold_leaf', 'carbon_black']},
            'blue': {'peak': 0.8, 'peak_wavelength': 470, 'variance': 0.08,
                    'possible_pigments': ['azurite', 'lapis_lazuli', 'synthetic_blue', 'indigo'],
                    'excluded_pigments': ['yellow_ochre', 'orpiment', 'gamboge']},
            'green': {'peak': 0.75, 'peak_wavelength': 530, 'variance': 0.09,
                     'possible_pigments': ['atacamite', 'malachite'],
                     'excluded_pigments': ['cinnabar', 'vermilion', 'carbon_black']},
            'yellow': {'peak': 0.88, 'peak_wavelength': 580, 'variance': 0.06,
                      'possible_pigments': ['yellow_ochre', 'orpiment', 'arsenic_sulfide', 
                                           'gamboge', 'phellodendron'],
                      'excluded_pigments': ['carbon_black', 'indigo']},
            'black': {'peak': 0.05, 'peak_wavelength': None, 'variance': 0.01,
                     'possible_pigments': ['carbon_black'],
                     'excluded_pigments': ['lead_white', 'calcium_carbonate', 'gold_leaf']},
            'brown': {'peak': 0.45, 'peak_wavelength': 600, 'variance': 0.15,
                     'possible_pigments': ['brown_alteration', 'hematite', 'yellow_ochre'],
                     'excluded_pigments': ['synthetic_blue', 'lapis_lazuli']},
            'gold': {'peak': 0.85, 'peak_wavelength': 590, 'variance': 0.03,
                    'possible_pigments': ['gold_leaf', 'orpiment'],
                    'excluded_pigments': ['carbon_black', 'indigo', 'azurite']},
            'gray': {'peak': 0.50, 'peak_wavelength': None, 'variance': 0.10,
                    'possible_pigments': ['gray_mixture', 'carbon_black', 'lead_white'],
                    'excluded_pigments': []},
            'purple': {'peak': 0.52, 'peak_wavelength': 440, 'variance': 0.08,
                      'possible_pigments': ['lac', 'indigo', 'cinnabar'],
                      'excluded_pigments': ['yellow_ochre', 'orpiment']},
            'orange': {'peak': 0.88, 'peak_wavelength': 620, 'variance': 0.10,
                      'possible_pigments': ['realgar', 'lead_oxide', 'red_ochre', 'orpiment'],
                      'excluded_pigments': ['azurite', 'lapis_lazuli', 'carbon_black']}
        }.items():
            spectral_values = self._generate_spectral_curve(
                data['peak'], data['peak_wavelength'], data['variance']
            )

            rgb_values = self._spectral_to_rgb(spectral_values)

            profiles[color] = {
                'spectral_values': spectral_values,
                'reflectance_peak': rgb_values, 
                'spectral_variance': data['variance'],
                'possible_pigments': data['possible_pigments'],
                'excluded_pigments': data['excluded_pigments'],
                'fluorescence': self._get_fluorescence_data(color),
                'brdf_params': self._get_default_brdf_params(color)
            }
            
        return profiles
    
    def _init_pigment_embeddings(self):
        """Initialize pigment embeddings using spectral and physical properties"""
        with torch.no_grad():
            for i in range(min(self.num_pigments, len(self.pigment_db.pigments))):
                pigment = self.pigment_db.pigments[i]

                try:
                    _, reflectance = self.pigment_db.get_spectral_reflectance(i)

                    if len(reflectance) != self.color_feat_dim:
                        reflectance_resampled = np.interp(
                            np.linspace(0, 1, self.color_feat_dim),
                            np.linspace(0, 1, len(reflectance)),
                            reflectance
                        )
                    else:
                        reflectance_resampled = reflectance

                    self.pigment_embedding.weight.data[i, :len(reflectance_resampled)] = torch.tensor(
                        reflectance_resampled, dtype=torch.float32
                    )
                    remaining_dims = self.color_feat_dim - len(reflectance_resampled)
                    if remaining_dims > 0:
                        if "reflectance" in pigment:
                            rgb = pigment["reflectance"]
                            start_idx = len(reflectance_resampled)
                            end_idx = min(start_idx + 3, self.color_feat_dim)
                            self.pigment_embedding.weight.data[i, start_idx:end_idx] = torch.tensor(
                                rgb[:end_idx-start_idx], dtype=torch.float32
                            )

                        if "roughness" in pigment and len(reflectance_resampled) + 4 <= self.color_feat_dim:
                            self.pigment_embedding.weight.data[i, len(reflectance_resampled)+3] = pigment["roughness"]

                        if "particle_size" in pigment and len(reflectance_resampled) + 5 <= self.color_feat_dim:
                            size = min(1.0, pigment["particle_size"]["mean"] / 10.0)  # Normalize to [0,1]
                            self.pigment_embedding.weight.data[i, len(reflectance_resampled)+4] = size
                except:
                    nn.init.normal_(self.pigment_embedding.weight.data[i], 0.0, 0.02)
    
    def _generate_spectral_curve(self, peak_reflectance, peak_wavelength, variance):
        """
        Generate a realistic spectral curve for a color category.
        
        Args:
            peak_reflectance: Maximum reflectance value
            peak_wavelength: Wavelength of maximum reflectance (None for flat spectra)
            variance: Spectral variance/width
            
        Returns:
            Spectral reflectance array covering 400-700nm range
        """
        wavelengths = np.linspace(400, 700, self.spectral_bands)
        if peak_wavelength is None:
            return np.ones(self.spectral_bands) * peak_reflectance

        spectral_values = np.zeros(self.spectral_bands)
        for i, wl in enumerate(wavelengths):
            if wl <= peak_wavelength:
                width = variance * 100 * 0.7
            else:
                width = variance * 100 * 1.3
            spectral_values[i] = peak_reflectance * np.exp(-((wl - peak_wavelength)**2) / (2 * width**2))

        if peak_wavelength == 440:  # Purple
            red_peak = 640
            red_strength = 0.4
            for i, wl in enumerate(wavelengths):
                red_contribution = red_strength * peak_reflectance * np.exp(-((wl - red_peak)**2) / (2 * (40**2)))
                spectral_values[i] += red_contribution

        if spectral_values.max() > 0:
            spectral_values = spectral_values * (peak_reflectance / spectral_values.max())
            
        return spectral_values
    
    def _get_fluorescence_data(self, color):
        """Get fluorescence parameters for different color categories"""
        fluorescence = {
            'active': False,
            'excitation_peak': 0,
            'emission_peak': 0,
            'quantum_yield': 0
        }

        if color == 'white':
            fluorescence = {
                'active': True,
                'excitation_peak': 360,  # UV excitation
                'emission_peak': 440,    # Blue emission
                'quantum_yield': 0.05    # Moderate yield
            }
        elif color == 'blue' or color == 'purple':
            fluorescence = {
                'active': True,
                'excitation_peak': 380,
                'emission_peak': 420,
                'quantum_yield': 0.1
            }
        elif color == 'yellow':
            fluorescence = {
                'active': True,
                'excitation_peak': 420,
                'emission_peak': 520,
                'quantum_yield': 0.15
            }
            
        return fluorescence
    
    def _get_default_brdf_params(self, color):
        """Get default BRDF parameters based on color category"""
        params = {
            'roughness': 0.5,
            'specular': 0.5,
            'metallic': 0.0,
            'anisotropy': 0.0,
            'subsurface': 0.0
        }

        if color == 'white':
            params['roughness'] = 0.3
            params['specular'] = 0.4
            params['subsurface'] = 0.4  
        elif color == 'black':
            params['roughness'] = 0.6
            params['specular'] = 0.1
        elif color == 'gold':
            params['roughness'] = 0.1
            params['specular'] = 0.9
            params['metallic'] = 0.9  
        elif color == 'red':
            params['roughness'] = 0.4
            params['specular'] = 0.3
            params['subsurface'] = 0.3  
        elif color == 'blue':
            params['roughness'] = 0.45
            params['specular'] = 0.3
            params['subsurface'] = 0.2
            
        return params
    
    def _init_microstructure_params(self):
        """Initialize microstructure parameters for different pigment types"""
        return {
            'mineral': {
                'porosity': 0.3,
                'facet_distribution': 'ggx',
                'correlation_length': 5.0,
                'rms_height': 0.8
            },
            'metal': {
                'porosity': 0.1,
                'facet_distribution': 'beckmann',
                'correlation_length': 2.0,
                'rms_height': 0.3
            },
            'organic': {
                'porosity': 0.5,
                'facet_distribution': 'ggx',
                'correlation_length': 8.0,
                'rms_height': 1.2
            },
            'clay': {
                'porosity': 0.6,
                'facet_distribution': 'ggx',
                'correlation_length': 10.0,
                'rms_height': 2.0
            }
        }
    
    def _init_pigment_name_map(self):
        """Initialize mapping from pigment names to their IDs"""
        self.pigment_name_to_id = {}

    def _update_pigment_name_map(self):
        """Update mapping from pigment names to their IDs"""
        for color, profile in self.spectral_profiles.items():
            for pigment_name in profile['possible_pigments']:
                if pigment_name in self.pigment_db.name_to_id:
                    self.pigment_name_to_id[pigment_name] = self.pigment_db.name_to_id[pigment_name]
    
    def _init_spectral_profiles(self):
        """Initialize spectral profiles with full hyperspectral data"""
        profiles = {}
        for color, data in {
            'white': {'peak': 0.92, 'peak_wavelength': None, 'variance': 0.02,
                    'possible_pigments': ['kaolin', 'calcium_carbonate', 'gypsum', 'lead_white',
                                        'quartz', 'lead_chloride', 'muscovite', 'talc', 
                                        'anhydrite', 'lead_sulfate'],
                    'excluded_pigments': []},
            'red': {'peak': 0.85, 'peak_wavelength': 650, 'variance': 0.12,
                'possible_pigments': ['cinnabar', 'vermilion', 'hematite', 'lead_oxide', 
                                        'realgar', 'cinnabar_variant', 'red_ochre', 'lac'],
                'excluded_pigments': ['gold_leaf', 'carbon_black']},
            'blue': {'peak': 0.8, 'peak_wavelength': 470, 'variance': 0.08,
                    'possible_pigments': ['azurite', 'lapis_lazuli', 'synthetic_blue', 'indigo'],
                    'excluded_pigments': ['yellow_ochre', 'orpiment', 'gamboge']},
            'green': {'peak': 0.75, 'peak_wavelength': 530, 'variance': 0.09,
                    'possible_pigments': ['atacamite', 'malachite'],
                    'excluded_pigments': ['cinnabar', 'vermilion', 'carbon_black']},
            'yellow': {'peak': 0.88, 'peak_wavelength': 580, 'variance': 0.06,
                    'possible_pigments': ['yellow_ochre', 'orpiment', 'arsenic_sulfide', 
                                        'gamboge', 'phellodendron'],
                    'excluded_pigments': ['carbon_black', 'indigo']},
            'black': {'peak': 0.05, 'peak_wavelength': None, 'variance': 0.01,
                    'possible_pigments': ['carbon_black'],
                    'excluded_pigments': ['lead_white', 'calcium_carbonate', 'gold_leaf']},
            'brown': {'peak': 0.45, 'peak_wavelength': 600, 'variance': 0.15,
                    'possible_pigments': ['brown_alteration', 'hematite', 'yellow_ochre'],
                    'excluded_pigments': ['synthetic_blue', 'lapis_lazuli']},
            'gold': {'peak': 0.85, 'peak_wavelength': 590, 'variance': 0.03,
                    'possible_pigments': ['gold_leaf', 'orpiment'],
                    'excluded_pigments': ['carbon_black', 'indigo', 'azurite']},
            'gray': {'peak': 0.50, 'peak_wavelength': None, 'variance': 0.10,
                    'possible_pigments': ['gray_mixture', 'carbon_black', 'lead_white'],
                    'excluded_pigments': []},
            'purple': {'peak': 0.52, 'peak_wavelength': 440, 'variance': 0.08,
                    'possible_pigments': ['lac', 'indigo', 'cinnabar'],
                    'excluded_pigments': ['yellow_ochre', 'orpiment']},
            'orange': {'peak': 0.88, 'peak_wavelength': 620, 'variance': 0.10,
                    'possible_pigments': ['realgar', 'lead_oxide', 'red_ochre', 'orpiment'],
                    'excluded_pigments': ['azurite', 'lapis_lazuli', 'carbon_black']}
        }.items():
            spectral_values = self._generate_spectral_curve(
                data['peak'], data['peak_wavelength'], data['variance']
            )
            
            rgb_values = self._spectral_to_rgb(spectral_values)

            profiles[color] = {
                'spectral_values': spectral_values,
                'reflectance_peak': rgb_values, 
                'spectral_variance': data['variance'],
                'possible_pigments': data['possible_pigments'],
                'excluded_pigments': data['excluded_pigments'],
                'fluorescence': self._get_fluorescence_data(color),
                'brdf_params': self._get_default_brdf_params(color)
            }
            
        self.device_initialized = False 
        return profiles

    def _extract_spectral_features(self, color_clusters):
        """
        Extract spectral features from color clusters.
        
        Args:
            color_clusters: List of color clusters from adaptive_color_clustering
            
        Returns:
            Tensor of spectral features
        """
        device = next(self.parameters()).device
        
        if not color_clusters:
            return None
                
        batch_size = len(color_clusters)
        batch_features = []
        for b in range(batch_size):
            clusters = color_clusters[b]
            if not clusters:
                batch_features.append(torch.zeros(self.color_feat_dim // 2, device=device))
                continue

            weighted_spectral = np.zeros(self.spectral_bands)
            total_weight = 0
            for cluster in clusters:
                if 'spectral_data' in cluster and 'values' in cluster['spectral_data']:
                    spectral_values = cluster['spectral_data']['values']
                    weight = cluster['percentage']
                    weighted_spectral += spectral_values * weight
                    total_weight += weight

            if total_weight > 0:
                weighted_spectral /= total_weight
                spectral_tensor = torch.tensor(weighted_spectral, dtype=torch.float32, device=device)
                spectral_features = self.spectral_encoder(spectral_tensor.unsqueeze(0)).squeeze(0)
            else:
                spectral_features = torch.zeros(self.color_feat_dim // 2, device=device)
                    
            batch_features.append(spectral_features)

        return torch.stack(batch_features)
        
    def _spectral_to_rgb(self, spectral_values):
        """
        Convert spectral values to RGB using CIE color matching functions.
        
        Args:
            spectral_values: Array of spectral reflectance values
            
        Returns:
            RGB values as 3-element array
        """
        if isinstance(spectral_values, np.ndarray):
            spectral_tensor = torch.tensor(spectral_values, dtype=torch.float32, device=self.device)
        else:
            spectral_tensor = spectral_values.to(self.device)

        if len(spectral_tensor.shape) > 1:
            spectral_tensor = spectral_tensor.squeeze()

        if not hasattr(self, 'device_initialized') or not self.device_initialized:
            self.cie_cmf = self.cie_cmf.to(self.device)

        rgb = torch.matmul(self.cie_cmf.to(self.device), spectral_tensor)

        if torch.max(rgb) > 0:
            rgb = rgb / torch.max(rgb)

        if torch.mean(spectral_tensor) < 0.1:
            rgb = rgb * (torch.mean(spectral_tensor) * 2)
                
        return rgb.detach().cpu().tolist()
    
    def _adaptive_color_clustering(self, image_tensor):
        """
        Adaptive color clustering using either MiniBatchKMeans or DBSCAN.
        """
        start_time = time.time()
        device = image_tensor.device

        if isinstance(image_tensor, torch.Tensor):
            if len(image_tensor.shape) == 4:
                batch_size, channels, height, width = image_tensor.shape
                image_np = image_tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
                image_np = image_np.detach().cpu().numpy()
            elif len(image_tensor.shape) == 2:
                image_np = image_tensor.unsqueeze(1).detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")
        else:
            image_np = image_tensor
                
        batch_size = image_np.shape[0]
        all_clusters = []
        for b in range(batch_size):
            img_data = image_np[b]
            
            if len(img_data.shape) == 3: 
                pixels = img_data.reshape(-1, img_data.shape[-1])
            else: 
                pixels = img_data
                    
            max_pixels = 100000  
            if pixels.shape[0] > max_pixels:
                indices = np.random.choice(pixels.shape[0], max_pixels, replace=False)
                pixel_sample = pixels[indices]
            else:
                pixel_sample = pixels

            if self.use_dbscan and self._should_use_dbscan(pixel_sample):
                clusters = self._dbscan_clustering(pixel_sample, pixels)
            else:
                batch_image_tensor = image_tensor[b] if isinstance(image_tensor, torch.Tensor) else None
                optimal_k = self._find_optimal_clusters(pixel_sample, batch_image_tensor)
                clusters = self._kmeans_clustering(pixel_sample, pixels, optimal_k)

            for cluster in clusters:
                if 'spectral_data' in cluster and 'values' in cluster['spectral_data']:
                    if isinstance(cluster['spectral_data']['values'], torch.Tensor):
                        cluster['spectral_data']['values'] = cluster['spectral_data']['values'].to(device)
            
            all_clusters.append(clusters)
            
        elapsed = time.time() - start_time
        if elapsed > 0.5: 
            print(f"Color clustering completed in {elapsed:.2f}s")
                
        return all_clusters
    
    def _should_use_dbscan(self, pixel_sample):
        """Determine if DBSCAN would be more appropriate than K-means"""
        if len(pixel_sample) < 100:
            return False

        color_var = np.var(pixel_sample, axis=0).sum()
        if len(pixel_sample) > 1000000:
            subsample = pixel_sample[np.random.choice(len(pixel_sample), 1000, replace=False)]
        else:
            subsample = pixel_sample

        pdist = cdist(subsample[:100], subsample[:100], 'euclidean')

        from scipy import stats
        kde = stats.gaussian_kde(pdist.flatten())
        x = np.linspace(0, np.max(pdist), 100)
        y = kde(x)

        peaks = []
        for i in range(1, len(y) - 1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                peaks.append(i)

        if len(peaks) > 3 or color_var < 0.01:
            return True
                
        return False
    
    def _dbscan_clustering(self, pixel_sample, all_pixels):
        """Use DBSCAN for clustering with automatic epsilon selection"""
        if len(pixel_sample) > 2000:
            subsample = pixel_sample[np.random.choice(len(pixel_sample), 2000, replace=False)]
        else:
            subsample = pixel_sample

        nbrs = NearestNeighbors(n_neighbors=5).fit(subsample)
        distances, indices = nbrs.kneighbors(subsample)
        k_dist = np.sort(distances[:, 4])
        x = np.arange(len(k_dist))
        y = k_dist
        
        try:
            tck = interpolate.splrep(x, y, s=len(x)/10)
            y_smooth = interpolate.splev(x, tck)
            y_smooth_d2 = np.gradient(np.gradient(y_smooth))
            maxima_idx = argrelextrema(y_smooth_d2, np.greater)[0]
            
            if len(maxima_idx) > 0:
                elbow_idx = maxima_idx[0]  
                epsilon = k_dist[elbow_idx]
            else:
                epsilon = k_dist[int(len(k_dist) * 0.1)]
        except:
            epsilon = k_dist[int(len(k_dist) * 0.1)]

        min_epsilon = 0.01
        epsilon = max(epsilon, min_epsilon)

        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=epsilon, min_samples=5)
        cluster_labels = dbscan.fit_predict(pixel_sample)
 
        if -1 in cluster_labels:
            outlier_indices = np.where(cluster_labels == -1)[0]
            valid_cluster_labels = np.unique(cluster_labels[cluster_labels != -1])
            
            if len(valid_cluster_labels) > 0:
                cluster_centers = []
                for label in valid_cluster_labels:
                    center = np.mean(pixel_sample[cluster_labels == label], axis=0)
                    cluster_centers.append(center)

                for idx in outlier_indices:
                    point = pixel_sample[idx]
                    distances = [np.linalg.norm(point - center) for center in cluster_centers]
                    closest_cluster = valid_cluster_labels[np.argmin(distances)]
                    cluster_labels[idx] = closest_cluster

        clusters = self._process_clusters(pixel_sample, all_pixels, cluster_labels)
        
        return clusters
    
    def _kmeans_clustering(self, pixel_sample, all_pixels, n_clusters):
        """Efficient K-means clustering with MiniBatchKMeans"""
        kmeans = MiniBatchKMeans(
            n_clusters = n_clusters, 
            random_state = 42, 
            batch_size = 256,
            max_iter = 120
        )
        kmeans.fit(pixel_sample)
        cluster_labels = kmeans.predict(pixel_sample)
        clusters = self._process_clusters(pixel_sample, all_pixels, cluster_labels)
        
        return clusters
    
    def _process_clusters(self, pixel_sample, all_pixels, cluster_labels):
        """Process clustering results into usable format with spectral data"""
        unique_labels = np.unique(cluster_labels)
        clusters = []
        for label in unique_labels:
            mask = (cluster_labels == label)
            if np.sum(mask) < max(5, len(pixel_sample) * self.cluster_threshold):
                continue

            center = np.mean(pixel_sample[mask], axis=0)
            percentage = np.sum(mask) / len(pixel_sample)
            adaptive_threshold = max(0.002, 0.5/(len(pixel_sample)**0.5))
            if percentage < adaptive_threshold:
                continue

            rgb = center
            color_name = self._rgb_to_color_name(rgb)
            cluster_data = {
                'center': center,
                'percentage': percentage,
                'color_name': color_name
            }
            
            if color_name in self.spectral_profiles:
                profile = self.spectral_profiles[color_name]
                adjusted_spectral = self._adjust_spectral_profile(
                    profile['spectral_values'], 
                    rgb
                )
                cluster_data['spectral_data'] = {
                    'values': adjusted_spectral,
                    'profile': profile
                }
                cluster_data['brdf_params'] = profile['brdf_params']
                cluster_data['fluorescence'] = profile['fluorescence']

                possible_pigments = []
                for p_name in profile['possible_pigments']:
                    if p_name in self.pigment_name_to_id:
                        possible_pigments.append(self.pigment_name_to_id[p_name])
                cluster_data['possible_pigments'] = possible_pigments
                
                excluded_pigments = []
                for p_name in profile['excluded_pigments']:
                    if p_name in self.pigment_name_to_id:
                        excluded_pigments.append(self.pigment_name_to_id[p_name])
                cluster_data['excluded_pigments'] = excluded_pigments
            else:
                spectral_values = self._rgb_to_spectral(rgb)
                cluster_data['spectral_data'] = {
                    'values': spectral_values,
                    'profile': {
                        'spectral_variance': 0.1
                    }
                }
                cluster_data['brdf_params'] = self._get_default_brdf_params('gray')
                cluster_data['fluorescence'] = {'active': False}
                cluster_data['possible_pigments'] = []
                cluster_data['excluded_pigments'] = []
            
            clusters.append(cluster_data)

        clusters.sort(key=lambda x: x['percentage'], reverse=True)
        
        return clusters
    
    def _find_optimal_clusters(self, pixels, image_tensor=None, max_attempt=50):
        """
        Find optimal number of clusters using enhanced methods with texture analysis.
        """
        max_silhouette_samples = 1000000
        if pixels.shape[0] > max_silhouette_samples:
            indices = np.random.choice(pixels.shape[0], max_silhouette_samples, replace=False)
            silhouette_sample = pixels[indices]
        else:
            silhouette_sample = pixels

        texture_var = 0.0
        if image_tensor is not None and len(image_tensor.shape) == 4:
            try:
                batch_image = image_tensor[0].cpu().numpy()
                gray_image = 0.2989 * batch_image[0] + 0.5870 * batch_image[1] + 0.1140 * batch_image[2]
                sobel_edges = sobel(gray_image)
                texture_var = np.var(sobel_edges)
            except Exception as e:
                warnings.warn(f"Texture analysis failed: {e}")

        color_variance = np.var(pixels, axis=0).sum()
        combined_variance = color_variance + texture_var * 0.5
        if combined_variance < 0.01:
            return max(2, self.min_clusters - 1)

        k_range = range(self.min_clusters, min(self.max_clusters + 1, 15))
        actual_k_values = np.linspace(min(k_range), max(k_range), 
                                    min(max_attempt, len(k_range)), 
                                    dtype=int).tolist()
        
        actual_k_values = sorted(list(set(actual_k_values)))
        silhouette_scores = []
        ch_scores = []
        for k in actual_k_values:
            try:
                kmeans = MiniBatchKMeans(
                    n_clusters=k, 
                    random_state=42, 
                    batch_size=100,
                    max_iter=100
                )
                cluster_labels = kmeans.fit_predict(silhouette_sample)

                if len(np.unique(cluster_labels)) > 1:
                    try:
                        sil_score = silhouette_score(silhouette_sample, cluster_labels, 
                                                random_state=42, sample_size=min(500, len(silhouette_sample)))
                        ch_score = calinski_harabasz_score(silhouette_sample, cluster_labels)
                        silhouette_scores.append(sil_score)
                        ch_scores.append(ch_score)
                    except Exception as e:
                        warnings.warn(f"Error calculating metrics for k={k}: {e}")
                        silhouette_scores.append(-1)
                        ch_scores.append(0)
                else:
                    silhouette_scores.append(-1)
                    ch_scores.append(0)
            except Exception as e:
                warnings.warn(f"Error in cluster analysis for k={k}: {e}")
                silhouette_scores.append(-1)
                ch_scores.append(0)
                continue

        if len(silhouette_scores) > 0 and max(silhouette_scores) > 0:
            if max(ch_scores) > 0:
                norm_ch_scores = [score / max(ch_scores) for score in ch_scores]
            else:
                norm_ch_scores = [0] * len(ch_scores)

            combined_scores = [0.7 * sil + 0.3 * ch for sil, ch in zip(silhouette_scores, norm_ch_scores)]
            best_idx = np.argmax(combined_scores)
            best_k = actual_k_values[best_idx]
        else:
            variance_factor = min(2.0, combined_variance * 20)
            spread = self.max_clusters - self.min_clusters
            texture_bonus = int(texture_var * 20)
            best_k = int(self.min_clusters + spread * variance_factor) + texture_bonus
            best_k = min(best_k, self.max_clusters)
            
        return best_k
    
    def _rgb_to_color_name(self, rgb_values):
        """Map RGB values to predefined color categories using spectral profiles"""
        min_dist = float('inf')
        closest_color = 'other'
        
        for color_name, profile in self.spectral_profiles.items():
            if 'reflectance_peak' in profile:
                peak = profile['reflectance_peak']
            else:
                peak = self._spectral_to_rgb(profile['spectral_values'])
                
            variance = profile['spectral_variance']
            weighted_dist = np.sqrt(sum([(a - b)**2 / max(variance, 0.01) 
                                       for a, b in zip(rgb_values, peak)]))
            if weighted_dist < min_dist:
                min_dist = weighted_dist
                closest_color = color_name
                
        return closest_color
    
    def _rgb_to_spectral(self, rgb_values):
        """
        Convert RGB values to plausible spectral reflectance.
        """
        r, g, b = rgb_values
        wavelengths = np.linspace(400, 700, self.spectral_bands)
        
        # Create basic components for reconstruction
        # 1. Flat reflectance (uniform across spectrum)
        flat = np.ones_like(wavelengths)
        
        # 2. Blue reflector basis function (peak in blue region)
        blue_peak = 450
        blue_width = 40
        blue_reflector = np.exp(-((wavelengths - blue_peak) ** 2) / (2 * blue_width ** 2))
        
        # 3. Green reflector basis function (peak in green region)
        green_peak = 520
        green_width = 30
        green_reflector = np.exp(-((wavelengths - green_peak) ** 2) / (2 * green_width ** 2))
        
        # 4. Red reflector basis function (peak in red region)
        red_peak = 620
        red_width = 50
        red_reflector = 1.0 / (1.0 + np.exp(-(wavelengths - 580) / 20)) 
        
        # 5. Yellow reflector (reflects both red and green)
        yellow_reflector = 1.0 / (1.0 + np.exp(-(wavelengths - 500) / 25)) 
        
        # 6. Cyan reflector (reflects both blue and green)
        cyan_reflector = 1.0 - red_reflector
        
        # 7. Magenta reflector (reflects both blue and red)
        magenta_reflector = 1.0 - green_reflector
        
        # Gray component (affects all wavelengths)
        gray = min(r, g, b)
        
        # Chromatic components
        r_chroma = max(0, r - gray) 
        g_chroma = max(0, g - gray)
        b_chroma = max(0, b - gray)

        w_flat = gray
        w_red = r_chroma
        w_green = g_chroma
        w_blue = b_chroma
        w_yellow = min(r_chroma, g_chroma) * 0.8
        w_cyan = min(g_chroma, b_chroma) * 0.8
        w_magenta = min(r_chroma, b_chroma) * 0.8
        spectral = (w_flat * flat + 
                   w_blue * blue_reflector + 
                   w_green * green_reflector + 
                   w_red * red_reflector +
                   w_yellow * yellow_reflector +
                   w_cyan * cyan_reflector +
                   w_magenta * magenta_reflector)
        if np.max(spectral) > 0:
            spectral = spectral / np.max(spectral)

        min_reflection = min(0.05, 0.5 * min(r, g, b) + 0.02)
        spectral = np.maximum(spectral, min_reflection)
        spectral = np.clip(spectral, 0, 1)
        
        return spectral
    
    def _adjust_spectral_profile(self, base_profile, rgb_values):
        """
        Adjust spectral profile to better match RGB values.
        
        Args:
            base_profile: Base spectral profile for color category
            rgb_values: Target RGB values to match
            
        Returns:
            Adjusted spectral values
        """
        if isinstance(base_profile, list):
            base_profile = np.array(base_profile)
        
        base_rgb = self._spectral_to_rgb(base_profile)

        scale_factors = []
        for i in range(3):
            if base_rgb[i] > 0.01: 
                scale_factors.append(rgb_values[i] / base_rgb[i])
            else:
                scale_factors.append(1.0)

        adjusted_profile = base_profile.copy()
 
        wavelengths = np.linspace(400, 700, len(base_profile))
        red_range = (wavelengths >= 580)
        green_range = (wavelengths >= 490) & (wavelengths < 580)
        blue_range = (wavelengths < 490)

        adjusted_profile[red_range] *= scale_factors[0]
        adjusted_profile[green_range] *= scale_factors[1]
        adjusted_profile[blue_range] *= scale_factors[2]
        adjusted_profile = np.clip(adjusted_profile, 0, 1)
        
        return adjusted_profile
    
    def _generate_color_histogram(self, image_tensor):
        """
        Generate color histogram from image tensor.
        
        Args:
            image_tensor: Image tensor [B, C, H, W]
            
        Returns:
            Color histogram tensor [B, 3*256]
        """
        if len(image_tensor.shape) == 4: 
            batch_size, channels, height, width = image_tensor.shape
            pixels = image_tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        elif len(image_tensor.shape) == 3:  
            channels, height, width = image_tensor.shape
            pixels = image_tensor.permute(1, 2, 0).reshape(1, -1, channels)
        elif len(image_tensor.shape) == 2:  
            return None
        else:
            raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")
        
        batch_size = pixels.shape[0]
        histograms = []
        for b in range(batch_size):
            img_pixels = pixels[b] 
            histogram = torch.zeros(3 * 256, device=img_pixels.device)
            for c in range(3):
                channel_values = img_pixels[:, c] 
                scaled_values = (channel_values * 255).long()
                scaled_values = torch.clamp(scaled_values, 0, 255)
                if img_pixels.device.type == 'cuda':
                    scaled_values_cpu = scaled_values.cpu()
                    bin_counts = torch.bincount(scaled_values_cpu, minlength=256).float()
                    bin_counts = bin_counts.to(img_pixels.device)
                else:
                    bin_counts = torch.bincount(scaled_values, minlength=256).float()

                if bin_counts.sum() > 0:
                    bin_counts = bin_counts / bin_counts.sum()
                
                histogram[c * 256:(c + 1) * 256] = bin_counts
            histograms.append(histogram)

        return torch.stack(histograms)
    
    def _apply_ml_pigment_mapping(self, color_feat, possible_pigments=None):
        """
        Apply machine learning based mapping from color features to pigments.
        
        Args:
            color_feat: Color features [batch_size, color_feat_dim]
            possible_pigments: Optional list of possible pigment indices per batch
            
        Returns:
            Tensor of pigment probabilities [batch_size, num_pigments]
        """
        batch_size = color_feat.shape[0]
        device = color_feat.device
        all_embeddings = self.pigment_embedding.weight
        color_norm = F.normalize(color_feat, p=2, dim=1)
        pigment_norm = F.normalize(all_embeddings, p=2, dim=1)
        similarity = torch.mm(color_norm, pigment_norm.t())
        temperature = 0.1 
        probs = F.softmax(similarity / temperature, dim=1)

        if possible_pigments is not None:
            mask = torch.zeros((batch_size, self.num_pigments), device=device)
            
            for b in range(batch_size):
                if b < len(possible_pigments) and possible_pigments[b]:
                    mask[b, possible_pigments[b]] = 1.0
                else:
                    mask[b, :] = 1.0

            masked_probs = probs * mask
            row_sums = masked_probs.sum(dim=1, keepdim=True)
            row_sums = torch.clamp(row_sums, min=1e-10) 
            masked_probs = masked_probs / row_sums
            
            return masked_probs
        
        return probs
    
    def _compute_kubelka_munk_mixture(self, pigment_indices, mixing_ratios):
        """
        Compute spectral reflectance for pigment mixture using Kubelka-Munk theory.
        
        Args:
            pigment_indices: Indices of pigments [batch_size, n_pigments]
            mixing_ratios: Mixing ratios [batch_size, n_pigments]
            
        Returns:
            Spectral reflectance of mixture [batch_size, spectral_bands]
        """
        batch_size, n_pigments = pigment_indices.shape
        device = pigment_indices.device
        mixture_reflectance = torch.zeros((batch_size, self.spectral_bands), device=device)
        
        for b in range(batch_size):
            k_over_s_values = []
            for i in range(n_pigments):
                pigment_id = pigment_indices[b, i].item()
                
                try:
                    k_s_data = self.pigment_db.get_absorption_and_scattering(pigment_id)
                    k_over_s = np.array(k_s_data["k_values"]) / np.array(k_s_data["s_values"])
                    if len(k_over_s) != self.spectral_bands:
                        k_over_s = np.interp(
                            np.linspace(0, 1, self.spectral_bands),
                            np.linspace(0, 1, len(k_over_s)),
                            k_over_s
                        )
                except:
                    try:
                        wavelengths, reflectance = self.pigment_db.get_spectral_reflectance(pigment_id)
                        if len(reflectance) != self.spectral_bands:
                            reflectance = np.interp(
                                np.linspace(400, 700, self.spectral_bands),
                                wavelengths,
                                reflectance
                            )
                        reflectance = np.clip(reflectance, 0.001, 0.999) 
                        k_over_s = ((1 - reflectance) ** 2) / (2 * reflectance)
                    except:
                        k_over_s = np.ones(self.spectral_bands) * 0.5
                    
                k_over_s_values.append(k_over_s)
            
            mixed_k_over_s = np.zeros(self.spectral_bands)
            total_ratio = 0
            for i in range(n_pigments):
                ratio = mixing_ratios[b, i].item()
                if ratio > 0:
                    mixed_k_over_s += k_over_s_values[i] * ratio
                    total_ratio += ratio

            if total_ratio > 0:
                mixed_k_over_s /= total_ratio

            mixed_reflectance = 1 + mixed_k_over_s - np.sqrt(mixed_k_over_s**2 + 2*mixed_k_over_s)
            mixed_reflectance = np.clip(mixed_reflectance, 0, 1)
            mixture_reflectance[b] = torch.tensor(mixed_reflectance, device=device)
        
        return mixture_reflectance
    
    def _enhance_pigment_mixture(self, base_probs, color_clusters=None):
        """
        Enhance pigment mixture probabilities using color clusters.
        
        Args:
            base_probs: Base pigment probabilities [batch_size, num_pigments]
            color_clusters: Optional color cluster data for guidance
            
        Returns:
            Enhanced pigment distribution
        """
        batch_size = base_probs.shape[0]
        device = base_probs.device

        if color_clusters is None:
            enhanced_probs = self.mixture_handler(
                torch.cat([base_probs, torch.pow(base_probs, 2)], dim=1)
            )
            return enhanced_probs
        enhanced_probs = torch.zeros_like(base_probs)
        for b in range(batch_size):
            clusters = color_clusters[b]
            if not clusters:
                enhanced_probs[b] = base_probs[b]
                continue

            weighted_pigments = defaultdict(float)
            total_weight = 0
            
            for cluster in clusters:
                weight = cluster['percentage']
                if 'possible_pigments' in cluster:
                    for pigment_id in cluster['possible_pigments']:
                        weighted_pigments[pigment_id] += weight
                
                total_weight += weight

            if weighted_pigments:
                for pigment_id in weighted_pigments:
                    weighted_pigments[pigment_id] /= total_weight
                pigment_mask = torch.zeros(self.num_pigments, device=device)
                
                for pigment_id, weight in weighted_pigments.items():
                    if 0 <= pigment_id < self.num_pigments:
                        pigment_mask[pigment_id] = weight
                blended = 0.7 * pigment_mask + 0.3 * base_probs[b]

                if torch.sum(blended) > 0:
                    blended = blended / torch.sum(blended)
                    refined = self.mixture_handler(
                        torch.cat([blended.unsqueeze(0), torch.pow(blended, 2).unsqueeze(0)], dim=1)
                    )
                    enhanced_probs[b] = refined.squeeze(0)
                else:
                    enhanced_probs[b] = base_probs[b]
            else:
                enhanced_probs[b] = base_probs[b]
        
        return enhanced_probs
    
    def _estimate_brdf_parameters(self, pigment_probs, color_features):
        """
        Estimate BRDF parameters for physically-based rendering.
        
        Args:
            pigment_probs: Pigment probability distribution [batch_size, num_pigments]
            color_features: Color feature vectors [batch_size, color_feat_dim]
            
        Returns:
            BRDF parameters [batch_size, 5]
        """
        combined_features = torch.cat([pigment_probs, color_features], dim=1)
        params = self.brdf_estimator(combined_features)
        
        return {
            'roughness': params[:, 0],
            'specular': params[:, 1],
            'metallic': params[:, 2],
            'anisotropy': params[:, 3],
            'subsurface': params[:, 4]
        }

    def _estimate_fluorescence_parameters(self, pigment_probs, color_features):
        """
        Estimate fluorescence parameters for physically-based rendering.
        
        Args:
            pigment_probs: Pigment probability distribution [batch_size, num_pigments]
            color_features: Color feature vectors [batch_size, color_feat_dim]
            
        Returns:
            Fluorescence parameters 
        """
        combined_features = torch.cat([pigment_probs, color_features], dim=1)
        params = self.fluorescence_estimator(combined_features)
        emission_strength = params[:, 0] 
        peak_shift = params[:, 1] * 50 - 25 
        bandwidth = params[:, 2] * 40 + 10 
        
        return {
            'emission_strength': emission_strength,
            'peak_shift': peak_shift,
            'bandwidth': bandwidth,
            'active': emission_strength > 0.05 
        }
    
    def _apply_cluster_constraints(self, base_probs, all_clusters):
        """
        Apply pigment constraints based on color clusters.
        """
        device = base_probs.device
        batch_size = base_probs.size(0)
        constrained_probs = torch.zeros_like(base_probs)
        
        for b in range(batch_size):
            clusters = all_clusters[b]
            if not clusters:
                constrained_probs[b] = base_probs[b]
                continue

            pigment_weights = torch.zeros(self.num_pigments, device=device)
            global_mask = torch.zeros(self.num_pigments, device=device)
            for cluster in clusters:
                percentage = cluster['percentage']
                possible_pigments = cluster.get('possible_pigments', [])
                if possible_pigments:
                    cluster_mask = torch.zeros_like(global_mask)
                    cluster_mask[possible_pigments] = 1.0
                    global_mask += cluster_mask * percentage
                else:
                    global_mask += 0.2 * percentage

            if global_mask.max() > 0:
                global_mask = global_mask / global_mask.max()

            for cluster in clusters:
                percentage = cluster['percentage']
                possible_pigments = cluster.get('possible_pigments', [])
                excluded_pigments = cluster.get('excluded_pigments', [])
                mask = torch.zeros(self.num_pigments, device=device)
                
                if possible_pigments:
                    mask[possible_pigments] = 1.0
                else:
                    mask = torch.ones_like(mask) * 0.5

                if excluded_pigments:
                    mask[excluded_pigments] = 0.0

                mask = mask * torch.sqrt(global_mask) 
                cluster_probs = base_probs[b] * mask
                if cluster_probs.sum() > 0:
                    cluster_probs = cluster_probs / cluster_probs.sum()
                pigment_weights += cluster_probs * percentage

            if pigment_weights.sum() > 0:
                constrained_probs[b] = pigment_weights / pigment_weights.sum()
            else:
                constrained_probs[b] = base_probs[b]
        enhanced_probs = self._enhance_pigment_mixture(constrained_probs)
        
        return enhanced_probs
    
    def forward(self, features, image_tensor=None, color_histogram=None):
        """
        Predict physically realistic pigment distributions.
        
        Args:
            features: Features from image encoder [batch_size, feat_dim]
            image_tensor: Optional raw image tensor for color analysis [batch_size, C, H, W]
            color_histogram: Optional pre-computed color histogram
            
        Returns:
            Dictionary with pigment probabilities and color clustering information
        """
        device = features.device
        batch_size = features.shape[0]
        if features.size(1) != self.swin_feat_dim:
            if len(features.shape) == 4: 
                features = F.adaptive_avg_pool2d(features, 1).reshape(features.size(0), -1)
            if features.size(1) == 3: 
                features = self.feature_adapter(features)
            else:
                if len(features.shape) == 2:  
                    features_unsqueezed = features.unsqueeze(-1)
                    pooled_features = F.adaptive_avg_pool1d(features_unsqueezed, self.swin_feat_dim)
                    features = pooled_features.squeeze(-1)
                else:
                    projection = nn.Linear(features.size(1), self.swin_feat_dim, device=device)
                    features = projection(features)

        if color_histogram is None and image_tensor is not None:
            color_histogram = self._generate_color_histogram(image_tensor)

        if color_histogram is None or color_histogram.size(1) != 3*256:
            print("Creating placeholder color histogram")
            color_histogram = torch.zeros((batch_size, 3*256), device=device)
            color_histogram = color_histogram + torch.rand_like(color_histogram) * 0.1

        color_feat = self.color_encoder(color_histogram)
        color_clusters = None
        if image_tensor is not None:
            color_clusters = self._adaptive_color_clustering(image_tensor)
            spectral_features = self._extract_spectral_features(color_clusters)
            if spectral_features is not None:
                color_feat = torch.cat([
                    color_feat, 
                    spectral_features.to(device)
                ], dim=1)

        joint_feat = torch.cat([features, color_feat], dim=1)
        pigment_logits = self.joint_mlp(joint_feat)
        self.pigment_prior = self.pigment_prior.to(device)
        self.style_modulation = self.style_modulation.to(device)
        prior_weights = self.pigment_prior * self.style_modulation
        weighted_logits = pigment_logits * prior_weights
        base_pigment_probs = F.softmax(weighted_logits, dim=1)
        if self.use_ml_mapping and color_clusters is not None:
            possible_pigments = []
            for clusters in color_clusters:
                cluster_pigments = set()
                for cluster in clusters:
                    if 'possible_pigments' in cluster:
                        cluster_pigments.update(cluster['possible_pigments'])
                possible_pigments.append(list(cluster_pigments))

            ml_probs = self._apply_ml_pigment_mapping(color_feat, possible_pigments)
            base_pigment_probs = 0.7 * ml_probs + 0.3 * base_pigment_probs

        constrained_probs = base_pigment_probs.clone()
        if color_clusters is not None:
            constrained_probs = self._apply_cluster_constraints(base_pigment_probs, color_clusters)

        brdf_params = self._estimate_brdf_parameters(constrained_probs, color_feat)
        fluorescence_params = self._estimate_fluorescence_parameters(constrained_probs, color_feat)
        k = 7 
        top_probs, top_indices = torch.topk(constrained_probs, k, dim=1)
        top_ratios = top_probs / top_probs.sum(dim=1, keepdim=True)
        spectral_data = self._compute_kubelka_munk_mixture(top_indices, top_ratios)
        
        return {
            'pigment_probs': constrained_probs,
            'base_probs': base_pigment_probs,
            'color_clusters': color_clusters,
            'brdf_params': brdf_params,
            'fluorescence_params': fluorescence_params,
            'spectral_data': spectral_data,
            'top_pigments': {
                'indices': top_indices,
                'ratios': top_ratios
            }
        }

    def get_pigment_suggestions(self, image_tensor, n_suggestions=5):
        """
        Get pigment suggestions for an image with color analysis..
        
        Args:
            image_tensor: Image tensor [1, C, H, W]
            n_suggestions: Number of pigment suggestions to return
            
        Returns:
            Dictionary with pigment suggestions and explanations
        """
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        avg_color = F.adaptive_avg_pool2d(image_tensor, 1).squeeze(-1).squeeze(-1)
        color_histogram = self._generate_color_histogram(image_tensor)
        color_clusters = self._adaptive_color_clustering(image_tensor)
        color_feat = self.color_encoder(color_histogram)
        spectral_features = self._extract_spectral_features(color_clusters)

        if spectral_features is not None:
            color_feat = torch.cat([
                color_feat, 
                spectral_features.to(color_feat.device)
            ], dim=1)

        features = self.feature_adapter(avg_color)
        joint_feat = torch.cat([features, color_feat], dim=1)
        pigment_logits = self.joint_mlp(joint_feat)
        prior_weights = self.pigment_prior * self.style_modulation
        weighted_logits = pigment_logits * prior_weights
        base_probs = F.softmax(weighted_logits, dim=1)
        constrained_probs = self._apply_cluster_constraints(base_probs, color_clusters)
        top_probs, top_indices = torch.topk(constrained_probs[0], min(n_suggestions * 2, self.num_pigments))

        suggestions = []
        for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
            if idx >= len(self.pigment_db.pigments) or prob < 0.01:
                continue
            if len(suggestions) >= n_suggestions:
                break
                
            pigment_name = self.pigment_db.id_to_name.get(idx, f"Pigment {idx}")
            suitable_colors = []
            for color, profile in self.spectral_profiles.items():
                if pigment_name in profile['possible_pigments']:
                    suitable_colors.append(color)

            if suitable_colors:
                explanation = f"Suggested for {', '.join(suitable_colors)} tones in the image"
            else:
                explanation = "Historically accurate pigment for this color range"

            try:
                pigment_data = self.pigment_db.pigments[idx]
                if 'formula' in pigment_data:
                    explanation += f". Chemical composition: {pigment_data['formula']}"
                if 'particle_size' in pigment_data:
                    size = pigment_data['particle_size']['mean']
                    unit = pigment_data['particle_size'].get('unit', 'micron')
                    explanation += f". Particle size: {size} {unit}"
                if hasattr(self.pigment_db, 'get_particle_data'):
                    particle_data = self.pigment_db.get_particle_data(idx)
                    if 'microstructure' in particle_data:
                        ms = particle_data['microstructure']
                        if 'porosity' in ms:
                            explanation += f". Porosity: {ms['porosity']:.2f}"

                physical_props = {
                    'spectral_reflectance': self.pigment_db.get_spectral_reflectance(idx)[1],
                    'roughness': self.pigment_db.get_roughness(idx)
                }

                if 'color' in pigment_data and pigment_data['color'] in self.spectral_profiles:
                    physical_props['brdf'] = self.spectral_profiles[pigment_data['color']]['brdf_params']
                if 'color' in pigment_data and pigment_data['color'] in self.spectral_profiles:
                    fluor = self.spectral_profiles[pigment_data['color']]['fluorescence']
                    if fluor.get('active', False):
                        physical_props['fluorescence'] = fluor
                        explanation += ". Has fluorescent properties"
                        
            except Exception as e:
                warnings.warn(f"Error getting pigment data for {pigment_name}: {e}")
                physical_props = {}

            matching_clusters = []
            for cluster in color_clusters[0]:
                if 'possible_pigments' in cluster and idx in cluster['possible_pigments']:
                    cluster_rgb = cluster['center']
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(255 * cluster_rgb[0]),
                        int(255 * cluster_rgb[1]),
                        int(255 * cluster_rgb[2])
                    )
                    matching_clusters.append({
                        'color': hex_color,
                        'percentage': cluster['percentage']
                    })
                    
            suggestions.append({
                'rank': len(suggestions) + 1,
                'pigment_id': idx,
                'pigment_name': pigment_name,
                'probability': prob,
                'explanation': explanation,
                'physical_properties': physical_props,
                'matching_clusters': matching_clusters
            })

        if len(suggestions) < n_suggestions:
            for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
                if idx >= len(self.pigment_db.pigments) or any(s['pigment_id'] == idx for s in suggestions):
                    continue
                if len(suggestions) >= n_suggestions:
                    break

                pigment_name = self.pigment_db.id_to_name.get(idx, f"Pigment {idx}")
                explanation = "Alternative pigment option based on historical practices"
                
                suggestions.append({
                    'rank': len(suggestions) + 1,
                    'pigment_id': idx,
                    'pigment_name': pigment_name,
                    'probability': prob,
                    'explanation': explanation,
                    'physical_properties': {},
                    'matching_clusters': []
                })
        top_k = 7
        top_mixture_probs, top_mixture_indices = torch.topk(constrained_probs[0], top_k)
        top_mixture_ratios = top_mixture_probs / top_mixture_probs.sum()
        mixture_spectral = self._compute_kubelka_munk_mixture(
            top_mixture_indices.unsqueeze(0), 
            top_mixture_ratios.unsqueeze(0)
        )
            
        return {
            'suggestions': suggestions,
            'color_clusters': color_clusters[0],  # First image in batch
            'spectral_data': mixture_spectral.cpu().numpy()[0]
        }
    
    def analyze_image_pigments(self, image, features=None):
        """
        Analyze an image for pigments
        
        Args:
            image: Input image tensor [B, C, H, W]
            features: Optional pre-extracted features
            
        Returns:
            Dictionary with pigment analysis
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        if features is None:
            features = F.adaptive_avg_pool2d(image, 1).squeeze(-1).squeeze(-1)
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1).reshape(features.size(0), -1)
        elif len(features.shape) == 3: 
            features = torch.mean(features, dim=2)

        try:
            pigment_data = self.forward(features, image)
        except RuntimeError as e:
            print(f"Warning: Forward method failed with error: {e}")
            print("Using fallback approach...")
            if not hasattr(self, "_projection_fallback") or self._projection_fallback.in_features != features.shape[1]:
                self._projection_fallback = nn.Linear(
                    features.shape[1], 
                    self.swin_feat_dim,
                    device=features.device
                )

            projected_features = self._projection_fallback(features)
            pigment_data = self.forward(projected_features, image)
        suggestions = self.get_pigment_suggestions(image)
        
        return {
            'pigment_probabilities': pigment_data['pigment_probs'],
            'color_clusters': pigment_data['color_clusters'],
            'pigment_suggestions': suggestions['suggestions'],
            'brdf_params': pigment_data['brdf_params'],
            'fluorescence_params': pigment_data['fluorescence_params'],
            'spectral_data': pigment_data['spectral_data'],
            'top_pigments': pigment_data['top_pigments']
        }
