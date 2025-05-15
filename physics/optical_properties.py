"""
Optical properties module for Dunhuang mural pigments.

This module provides physically accurate optical properties for Dunhuang mural pigments.
It includes absorption, scattering, reflectance, and surface properties predictions.
"""
import torch
import torch.nn as nn
import numpy as np
import warnings

from pigment_database import DunhuangPigmentDB
from themodynamicval import ThermodynamicValidator

# ===== Optical Properties Module =====
class OpticalProperties(nn.Module):
    """
    Optical properties module for Dunhuang mural pigments.
    """
    def __init__(self, num_pigments=35, pigment_embedding_dim=64, spectral_bands=31, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_pigments = num_pigments
        self.pigment_embedding_dim = pigment_embedding_dim
        self.spectral_bands = spectral_bands
        self.pigment_db = DunhuangPigmentDB()
        self.wavelengths = torch.linspace(400, 700, spectral_bands, device=self.device)
        self.pigment_embedding = nn.Embedding(num_pigments, pigment_embedding_dim)

        self.absorption_coef_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, spectral_bands),
            nn.Softplus()
        )

        self.scattering_coef_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim, 128),
            nn.LayerNorm(128), 
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),  
            nn.LeakyReLU(0.2),
            nn.Linear(64, spectral_bands),
            nn.Softplus()
        )
        
        self.reflectance_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, spectral_bands),
            nn.Sigmoid()
        )

        self.surface_props_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim, 64),
            nn.LayerNorm(64),  
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 5),
            nn.Sigmoid()
        )

        self.fluorescence_pred = nn.Sequential(
            nn.Linear(pigment_embedding_dim, 64),
            nn.LayerNorm(64), 
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32), 
            nn.LeakyReLU(0.2),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )

        spectral_to_rgb = self._init_cie_color_matching()
        self.register_buffer('spectral_to_rgb', spectral_to_rgb.to(self.device))
        self._initialize_from_dunhuang_pigments()
        self.to(self.device)
        
    def _init_cie_color_matching(self):
        """Initialize spectral to RGB conversion using actual CIE 1931 color matching functions."""
        wavelengths_nm = np.linspace(400, 700, self.spectral_bands)
        
        # Standard CIE 1931 XYZ color matching functions (x̄, ȳ, z̄)
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
        
        # Interpolate to match our spectral resolution
        x_function = np.interp(wavelengths_nm, cie_wavelengths, x_bar_values)
        y_function = np.interp(wavelengths_nm, cie_wavelengths, y_bar_values)
        z_function = np.interp(wavelengths_nm, cie_wavelengths, z_bar_values)
        
        # XYZ to RGB conversion matrix (sRGB standard with D65 white point)
        xyz_to_rgb = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ])
        
        # Combine XYZ curves with conversion matrix to get RGB curves
        cmf_xyz = np.stack([x_function, y_function, z_function], axis=0)
        cmf_rgb = np.dot(xyz_to_rgb, cmf_xyz)
        
        # Ensure no negative values in the color matching functions
        cmf_rgb = np.maximum(0, cmf_rgb)
        
        # Normalize to maintain proper intensity scaling
        if np.max(cmf_rgb) > 0:
            cmf_rgb = cmf_rgb / np.max(cmf_rgb)
        
        return torch.tensor(cmf_rgb, dtype=torch.float32)
    
    def _initialize_from_dunhuang_pigments(self):
        """
        Initialize with physically-based values from Dunhuang pigment data.
        """
        nn.init.xavier_normal_(self.pigment_embedding.weight)
        
        with torch.no_grad():
            for i in range(min(self.num_pigments, len(self.pigment_db.pigments))):
                pigment = self.pigment_db.pigments[i]
                if "reflectance" in pigment:
                    rgb_reflectance = torch.tensor(pigment["reflectance"])
                    self.pigment_embedding.weight.data[i, 0:3] = rgb_reflectance
                    r, g, b = rgb_reflectance
                    for j in range(3, min(20, self.pigment_embedding_dim)):
                        if j % 3 == 0:
                            self.pigment_embedding.weight.data[i, j] = r
                        elif j % 3 == 1:
                            self.pigment_embedding.weight.data[i, j] = g
                        else:
                            self.pigment_embedding.weight.data[i, j] = b

                    absorption_estimate = 1.0 - rgb_reflectance.mean()
                    scattering_estimate = rgb_reflectance.mean()
                    
                    idx = min(self.pigment_embedding_dim - 2, 20)
                    self.pigment_embedding.weight.data[i, idx] = absorption_estimate
                    self.pigment_embedding.weight.data[i, idx+1] = scattering_estimate

                try:
                    if hasattr(self.pigment_db, 'get_spectral_reflectance'):
                        wavelengths, reflectance = self.pigment_db.get_spectral_reflectance(i)
                        if len(reflectance) != self.spectral_bands:
                            reflectance = np.interp(
                                self.wavelengths.cpu().numpy(),
                                wavelengths,
                                reflectance
                            )
                        for j, ref in enumerate(reflectance[:min(len(reflectance), self.pigment_embedding_dim // 2)]):
                            self.pigment_embedding.weight.data[i, j] = ref
                except Exception as e:
                    warnings.warn(f"Error loading spectral data for pigment {i}: {e}")

                if "roughness" in pigment:
                    idx = min(self.pigment_embedding_dim - 1, 22)
                    self.pigment_embedding.weight.data[i, idx] = pigment["roughness"]

                if "aging" in pigment:
                    if isinstance(pigment["aging"].get("yellowing", 0), dict):
                        self.pigment_embedding.weight.data[i, 23] = pigment["aging"]["yellowing"]["rate"]
                        self.pigment_embedding.weight.data[i, 24] = pigment["aging"]["darkening"]["rate"]
                        self.pigment_embedding.weight.data[i, 25] = pigment["aging"]["fading"]["rate"]
                    else:
                        self.pigment_embedding.weight.data[i, 23] = pigment["aging"]["yellowing"]
                        self.pigment_embedding.weight.data[i, 24] = pigment["aging"]["darkening"]
                        self.pigment_embedding.weight.data[i, 25] = pigment["aging"]["fading"]

                if "particle_size" in pigment:
                    mean_size = min(10.0, pigment["particle_size"]["mean"]) 
                    std_size = min(3.0, pigment["particle_size"]["std"])    
                    self.pigment_embedding.weight.data[i, 26] = mean_size / 10.0 
                    self.pigment_embedding.weight.data[i, 27] = std_size / 3.0   

                try:
                    if hasattr(self.pigment_db, 'get_particle_data'):
                        particle_data = self.pigment_db.get_particle_data(i)
                        if 'microstructure' in particle_data:
                            ms = particle_data['microstructure']
                            if 'porosity' in ms:
                                self.pigment_embedding.weight.data[i, 28] = ms['porosity']
                            if 'specific_surface_area' in ms:
                                self.pigment_embedding.weight.data[i, 29] = ms['specific_surface_area'] / 20.0 

                        if 'refractive_index' in particle_data:
                            ri = particle_data['refractive_index']
                            if isinstance(ri, dict) and 'real' in ri and 'imag' in ri:
                                real_index = ri['real']
                                imag_index = ri['imag']
                                self.pigment_embedding.weight.data[i, 30] = real_index
                                self.pigment_embedding.weight.data[i, 31] = imag_index
                except Exception as e:
                    pass 
    
    def _spectral_to_rgb_conversion(self, spectral_values):
        """
        Convert spectral reflectance values to RGB.
        """
        if isinstance(spectral_values, list):
            spectral_values = torch.tensor(spectral_values, device=self.device, dtype=torch.float32)
        elif isinstance(spectral_values, np.ndarray):
            spectral_values = torch.tensor(spectral_values, device=self.device, dtype=torch.float32)
        if spectral_values.dim() == 1:
            spectral_values = spectral_values.unsqueeze(0)

        spectral_values = torch.nan_to_num(spectral_values, nan=0.5)
        original_shape = spectral_values.shape
        spectral_flat = spectral_values.reshape(-1, self.spectral_bands)
        spectral_flat = spectral_flat.float()

        rgb_flat = torch.matmul(spectral_flat, self.spectral_to_rgb.T)
        d65_spectral = torch.ones_like(spectral_flat[0:1])
        d65_rgb = torch.matmul(d65_spectral, self.spectral_to_rgb.T)

        d65_rgb = torch.clamp(d65_rgb, min=1e-8)
        rgb_flat = rgb_flat / d65_rgb
        rgb_flat = torch.clamp(rgb_flat, 0.0, 1.0)
        rgb_flat = torch.nan_to_num(rgb_flat, nan=0.5)
        
        linear_mask = rgb_flat <= 0.0031308
        rgb_flat[linear_mask] = 12.92 * rgb_flat[linear_mask]
        rgb_flat[~linear_mask] = 1.055 * torch.pow(rgb_flat[~linear_mask], 1/2.4) - 0.055

        if len(original_shape) == 1:
            rgb_values = rgb_flat.squeeze(0)
        else:
            new_shape = original_shape[:-1] + (3,)
            rgb_values = rgb_flat.reshape(new_shape)
        
        return rgb_values
        
    def _kubelka_munk_mixing(self, absorption_coeffs, scattering_coeffs, concentrations=None):
        """
        Apply Kubelka-Munk theory for physically accurate pigment mixing with Saunderson correction.
        
        Args:
            absorption_coeffs: Absorption coefficients [batch_size, n_pigments, spectral_bands]
            scattering_coeffs: Scattering coefficients [batch_size, n_pigments, spectral_bands]
            concentrations: Relative concentrations [batch_size, n_pigments]
            
        Returns:
            Mixed spectral reflectance [batch_size, spectral_bands]
        """
        batch_size, n_pigments, _ = absorption_coeffs.shape
        if concentrations is None:
            concentrations = torch.ones(batch_size, n_pigments, device=absorption_coeffs.device)
            concentrations = concentrations / n_pigments
        concentrations = concentrations / (concentrations.sum(dim=1, keepdim=True) + 1e-8)
        conc_expanded = concentrations.unsqueeze(-1) 
        
        # Apply concentrations to K/S ratios
        k_over_s_values = []
        
        for i in range(n_pigments):
            # Calculate K/S ratio for each pigment
            k = absorption_coeffs[:, i, :]
            s = scattering_coeffs[:, i, :]
            k_over_s = k / (s + 1e-8)
            k_over_s_values.append(k_over_s)
        
        k_over_s_stack = torch.stack(k_over_s_values, dim=1)
        
        # Calculate weighted average of K/S ratios
        mixed_k_over_s = torch.sum(k_over_s_stack * conc_expanded, dim=1) 
        
        # Calculate internal reflectance using Kubelka-Munk equation
        r_infinity = 1.0 + mixed_k_over_s - torch.sqrt(mixed_k_over_s * (mixed_k_over_s + 2.0) + 1e-8)
        
        # Apply Saunderson correction for surface reflections
        # k1 = external surface reflection, k2 = internal surface reflection
        k1 = 0.04 
        k2 = 0.6   
        r_corrected = k1 + ((1.0 - k1) * (1.0 - k2) * r_infinity) / (1.0 - k2 * r_infinity + 1e-8)
        r_corrected = torch.clamp(r_corrected, 0.0, 1.0)
        
        return r_corrected
    
    def _compute_mie_scattering(self, wavelengths, particle_size, refractive_index):
        """
        Compute Mie scattering effects based on analytical approximations.
            
        Args:
            wavelengths: Light wavelengths
            particle_size: Mean particle size in microns
            refractive_index: Refractive index relative to medium (complex)
            
        Returns:
            Scattering efficiency, absorption efficiency, asymmetry parameter
        """
        wavelengths_um = wavelengths / 1000.0
        
        # Size parameter x = 2πr/λ
        size_parameter = 2 * np.pi * particle_size * torch.real(torch.tensor(refractive_index, dtype=torch.complex64)) / wavelengths_um
        scattering_efficiency = torch.zeros_like(wavelengths)
        absorption_efficiency = torch.zeros_like(wavelengths)
        asymmetry = torch.zeros_like(wavelengths)

        # Rayleigh regime (x << 1): small particles compared to wavelength
        rayleigh_mask = size_parameter < 0.5
        # Mie regime: comparable to wavelength
        mie_mask = (size_parameter >= 0.5) & (size_parameter < 10)
        # Geometric regime (x >> 1): large particles compared to wavelength
        geometric_mask = size_parameter >= 10
        
        # Rayleigh scattering formula
        if torch.any(rayleigh_mask):
            x = size_parameter[rayleigh_mask]
            m = torch.tensor(refractive_index, dtype=torch.complex64)
            m_squared = m * m
            scattering_efficiency[rayleigh_mask] = (8/3) * (x**4) * torch.abs((m_squared - 1)/(m_squared + 2))**2 * 5.0 
            absorption_efficiency[rayleigh_mask] = 4 * x * torch.imag(m)
            asymmetry[rayleigh_mask] = 0.0
                
        # Mie regime
        if torch.any(mie_mask):
            x = size_parameter[mie_mask]
            m = torch.tensor(refractive_index, dtype=torch.complex64)
            m_real = torch.real(m)
            m_imag = torch.imag(m)
            scattering_efficiency[mie_mask] = 2 - 4 * torch.exp(-0.7 * x) * torch.cos(x) / x + 2 * torch.exp(-0.7 * x) / (x**2)
            absorption_efficiency[mie_mask] = 1 - torch.exp(-4 * np.pi * m_imag * x)
            asymmetry[mie_mask] = (1 - torch.exp(-1.4 * x)) * (0.8 - 0.55 * torch.exp(-0.3 * x))
            
        # Geometric optics regime
        if torch.any(geometric_mask):
            m_imag = torch.imag(torch.tensor(refractive_index, dtype=torch.complex64))
            scattering_efficiency[geometric_mask] = 2.0
            absorption_efficiency[geometric_mask] = 1 - torch.exp(-4 * np.pi * m_imag)
            asymmetry[geometric_mask] = 0.85
                
        return scattering_efficiency, absorption_efficiency, asymmetry
    
    def _apply_particle_size_effects_vectorized(self, wavelengths, spectral_reflectance, particle_data):
        """Particle size effects using Mie theory."""
        batch_size = spectral_reflectance.shape[0]
        device = spectral_reflectance.device

        if isinstance(particle_data['mean'], (int, float)):
            mean_size = torch.full((batch_size,), particle_data['mean'], device=device)
        else:
            mean_size = particle_data['mean'].to(device)
        mean_size = mean_size.unsqueeze(1)

        spectral_reflectance = torch.clamp(spectral_reflectance, 1e-6, 0.999)
        K_over_S = (1 - spectral_reflectance)**2 / (2 * spectral_reflectance)

        modified_K_over_S = K_over_S.clone()
        for i in range(batch_size):
            size = mean_size[i].item() if hasattr(mean_size[i], 'item') else mean_size[i]
            refractive_index = particle_data.get('refractive_index', 1.5)
 
            scattering_eff, absorption_eff, asymmetry = self._compute_mie_scattering(
                wavelengths, size, refractive_index
            )
            
            scattering_factor = torch.tensor(scattering_eff, device=device) / torch.mean(torch.tensor(scattering_eff, device=device))
            absorption_factor = torch.tensor(absorption_eff, device=device) / torch.mean(torch.tensor(absorption_eff, device=device))

            size_factor = size / 5.0  
            simple_scaling = K_over_S[i] * (1.0 / torch.clamp(torch.tensor(size_factor, device=device), min=0.2))
            mie_scaling = K_over_S[i] * (absorption_factor / torch.clamp(scattering_factor, min=0.1))

            modified_K_over_S[i] = 0.3 * simple_scaling + 0.7 * mie_scaling

        modified_K_over_S = torch.nan_to_num(modified_K_over_S, nan=0.5) 
        modified_reflectance = 1 + modified_K_over_S - torch.sqrt(modified_K_over_S**2 + 2*modified_K_over_S)
        modified_reflectance = torch.clamp(modified_reflectance, 0.0, 1.0)
        
        return modified_reflectance
    
    def apply_spectral_aging(self, spectral_values, aged_spectral_values):
        """
        Apply pre-computed aging effects from ThermodynamicValidator to spectral values.
        
        Args:
            spectral_values: Original spectral values [batch_size, spectral_bands]
            aged_spectral_values: Aged spectral values from ThermodynamicValidator
            
        Returns:
            Tuple of (aged_rgb_values, aged_spectral_tensor)
        """
        if isinstance(aged_spectral_values, np.ndarray):
            aged_spectral_tensor = torch.tensor(aged_spectral_values, device=self.device, dtype=torch.float32)
        elif isinstance(aged_spectral_values, list):
            aged_spectral_tensor = torch.tensor(aged_spectral_values, device=self.device, dtype=torch.float32)
        else:
            aged_spectral_tensor = aged_spectral_values.to(self.device)

        aged_spectral_tensor = torch.nan_to_num(aged_spectral_tensor, nan=0.5)
        aged_rgb = self._spectral_to_rgb_conversion(aged_spectral_tensor)
        
        return aged_rgb, aged_spectral_tensor
    
    def _simulate_fluorescence(self, spectral_values, fluorescence_params):
        """
        Simulate fluorescence effects for pigments with fluorescent properties.
            
        Args:
            spectral_values: Original spectral reflectance
            fluorescence_params: Dict with fluorescence parameters
            
        Returns:
            Modified spectral reflectance with fluorescence effects
        """
        emission_strength = fluorescence_params['emission_strength']
        if emission_strength < 0.01:
            return spectral_values
        
        excitation_peak = 380 + fluorescence_params['excitation_peak'] * 70  # 380-450nm range
        emission_peak = 420 + fluorescence_params['emission_peak'] * 180     # 420-600nm range
        bandwidth = 10 + fluorescence_params['bandwidth'] * 40               # 10-50nm range

        device = spectral_values.device
        wavelengths = self.wavelengths.to(device)
        
        # Calculate excitation and emission profiles
        excitation_profile = torch.exp(-((wavelengths - excitation_peak) ** 2) / (2 * 15 ** 2))
        emission_profile = torch.exp(-((wavelengths - emission_peak) ** 2) / (2 * bandwidth ** 2))
        if excitation_profile.sum() > 0:
            excitation_profile = excitation_profile / excitation_profile.sum()
        if emission_profile.sum() > 0:
            emission_profile = emission_profile / emission_profile.sum()
        
        # Calculate absorbed energy from excitation wavelengths
        absorption = 1.0 - spectral_values
        absorbed_energy = (absorption * excitation_profile).sum()
        
        # Compute quantum yield with energy conservation (Lakowicz, 2006)
        quantum_yield = emission_strength
        
        # Energy conservation constraint - cannot emit more energy than absorbed
        stokes_loss_factor = excitation_peak / emission_peak 
        max_theoretical_yield = 0.9
        quantum_yield = torch.clamp(
            torch.tensor(quantum_yield * stokes_loss_factor, device=device), 
            0.0, 
            max_theoretical_yield
        )
        
        # Apply fluorescence emission based on absorbed energy with correct quantum yield
        fluorescence_contribution = emission_profile * absorbed_energy * quantum_yield
        
        # Add fluorescence to original reflection
        modified_spectral = torch.min(
            spectral_values + fluorescence_contribution,
            torch.ones_like(spectral_values)
        )
        
        return modified_spectral
    
    def _apply_brdf_parameters(self, rgb_values, surface_props):
        """
        Apply enhanced BRDF parameters for physically-based rendering.
        
        Args:
            rgb_values: Base RGB colors [batch_size, 3]
            surface_props: Surface properties [batch_size, 5]
            (roughness, metallic, specular, anisotropy, subsurface)
            
        Returns:
            Dictionary with BRDF parameters for rendering
        """
        if not torch.is_tensor(surface_props):
            surface_props = torch.tensor(surface_props, device=self.device, dtype=torch.float32)
        if surface_props.dim() == 1:
            surface_props = surface_props.unsqueeze(0) 
        if surface_props.size(1) < 5:
            padding = torch.zeros(surface_props.size(0), 5 - surface_props.size(1), 
                                device=surface_props.device)
            surface_props = torch.cat([surface_props, padding], dim=1)

        roughness = surface_props[:, 0:1]     # Surface roughness
        metallic = surface_props[:, 1:2]      # Metallic vs. dielectric
        specular = surface_props[:, 2:3]      # Specular reflection strength
        anisotropy = surface_props[:, 3:4]    # Directional variation
        subsurface = surface_props[:, 4:5]    # Subsurface scattering

        albedo = rgb_values
        
        # Metallic workflow: derive specular F0 from albedo and metallic
        f0_dielectric = torch.full_like(albedo, 0.04)
 
        metallic_expanded = metallic.expand(-1, 3) if metallic.size(1) == 1 else metallic
        f0 = torch.lerp(f0_dielectric, albedo, metallic_expanded)
        diffuse_color = albedo * (1.0 - metallic_expanded)

        roughness_x = roughness * (1.0 + anisotropy)
        roughness_y = roughness * (1.0 - anisotropy)
        roughness_x = torch.clamp(roughness_x, 0.01, 1.0)
        roughness_y = torch.clamp(roughness_y, 0.01, 1.0)
        
        alpha_x = roughness_x * roughness_x
        alpha_y = roughness_y * roughness_y
        
        return {
            'albedo': albedo,
            'roughness': roughness.mean() if torch.is_tensor(roughness) and roughness.numel() > 1 else roughness,
            'metallic': metallic.mean() if torch.is_tensor(metallic) and metallic.numel() > 1 else metallic,
            'specular': specular.mean() if torch.is_tensor(specular) and specular.numel() > 1 else specular,
            'f0': f0,
            'diffuse_color': diffuse_color, 
            'anisotropy': anisotropy.mean() if torch.is_tensor(anisotropy) and anisotropy.numel() > 1 else anisotropy,
            'subsurface': subsurface.mean() if torch.is_tensor(subsurface) and subsurface.numel() > 1 else subsurface,
            'alpha_x': alpha_x.mean() if torch.is_tensor(alpha_x) and alpha_x.numel() > 1 else alpha_x,
            'alpha_y': alpha_y.mean() if torch.is_tensor(alpha_y) and alpha_y.numel() > 1 else alpha_y
        }

    def _calculate_layered_reflectance(self, layer_properties):
        """
        Calculate reflectance of multiple pigment layers.
            
        Args:
            layer_properties: List of dictionaries with layer properties
            Each should contain:
                - spectral: Spectral reflectance of the layer
                - thickness: Layer thickness in microns
                - interfaces: Interface properties
                - refractive_index: Complex refractive index (optional)
            
        Returns:
            Effective spectral reflectance of the layered structure
        """
        if not layer_properties:
            return torch.ones(self.spectral_bands).to(self.wavelengths.device)
        
        bottom_reflectance = layer_properties[0]['spectral']
        for i in range(1, len(layer_properties)):
            current = layer_properties[i]
            spectral = current['spectral']
            thickness = current.get('thickness', 20.0) 
            interface = current.get('interface', {})
            effective_thickness = thickness * 2.0 

            if 'refractive_index' in current:
                ri = current['refractive_index']
                if isinstance(ri, dict) and 'real' in ri and 'imag' in ri:
                    real_part = ri['real']
                    imag_part = ri['imag']
                    if isinstance(imag_part, (int, float)):
                        imag_part = imag_part * 2.0
                    
                    # Calculate wavelength-dependent Fresnel reflection
                    # For each wavelength: R = ((n1-n2)/(n1+n2))^2 + ((k1-k2)/(k1+k2))^2
                    n1 = 1.0  
                    k1 = 0.0  

                    if isinstance(real_part, (list, np.ndarray, torch.Tensor)):
                        if len(real_part) == self.spectral_bands:
                            n2 = torch.tensor(real_part, device=self.wavelengths.device)
                        else:
                            n2 = torch.tensor(
                                np.interp(
                                    self.wavelengths.cpu().numpy(), 
                                    np.linspace(400, 700, len(real_part)), 
                                    real_part
                                ), 
                                device=self.wavelengths.device
                            )
                    else:
                        n2 = torch.full_like(self.wavelengths, real_part)

                    if isinstance(imag_part, (list, np.ndarray, torch.Tensor)):
                        if len(imag_part) == self.spectral_bands:
                            k2 = torch.tensor(imag_part, device=self.wavelengths.device)
                        else:
                            k2 = torch.tensor(
                                np.interp(
                                    self.wavelengths.cpu().numpy(), 
                                    np.linspace(400, 700, len(imag_part)), 
                                    imag_part
                                ), 
                                device=self.wavelengths.device
                            )
                    else:
                        k2 = torch.full_like(self.wavelengths, imag_part)
                    
                    # Calculate Fresnel reflection
                    n_term = ((n1 - n2) / (n1 + n2))**2
                    k_term = ((k1 - k2) / (k1 + k2 + 1e-8))**2 
                    internal_reflection = n_term + k_term
                else:
                    n = ri if isinstance(ri, (int, float)) else 1.5
                    internal_reflection = ((1.0 - n) / (1.0 + n))**2
            else:
                internal_reflection = 0.04  
            
            # Convert reflectance to K/S for both layers
            bottom_k_over_s = (1 - bottom_reflectance)**2 / (2 * bottom_reflectance.clamp(min=1e-6))
            current_k_over_s = (1 - spectral)**2 / (2 * spectral.clamp(min=1e-6))
            current_k_over_s = current_k_over_s * 1.5 
            
            # Calculate transmission factors based on layer thickness
            transmission = torch.exp(-effective_thickness * 0.1 * (1 + current_k_over_s))
            
            # Calculate effective reflectance with internal reflections
            numerator = spectral + (transmission * bottom_reflectance * transmission) / (
                1 - internal_reflection * bottom_reflectance)
            denominator = 1 + spectral * bottom_reflectance * internal_reflection
            bottom_reflectance = numerator / denominator.clamp(min=1e-6)
            bottom_reflectance = torch.clamp(bottom_reflectance, 0.0, 1.0)
        
        return bottom_reflectance
        
    def forward(self, pigment_indices, mixing_ratios=None, precomputed_aging=None, 
        particle_data=None, fluorescence_params=None, microstructure_params=None,
        layered_configuration=None):
        """
        Calculate physically-based optical properties for given pigments.
        
        Args:
            pigment_indices: Indices of pigments [batch_size] or [batch_size, n_pigments]
            mixing_ratios: Optional mixing ratios [batch_size, n_pigments]
            precomputed_aging: Optional precomputed aged spectral data
            particle_data: Optional particle size data dictionary
            fluorescence_params: Optional fluorescence parameters
            microstructure_params: Optional microstructure parameters
            layered_configuration: Optional configuration for layered rendering
            
        Returns:
            Dictionary with optical properties
        """
        original_shape = pigment_indices.shape
        batch_size = original_shape[0]
        is_multi_pigment = len(original_shape) > 1
        pigment_indices = torch.clamp(pigment_indices, 0, self.num_pigments - 1)
        if is_multi_pigment:
            n_pigments = original_shape[1]
            
            flat_indices = pigment_indices.reshape(-1)
            flat_embed = self.pigment_embedding(flat_indices)
            pigment_embed = flat_embed.reshape(batch_size, n_pigments, -1)
            flat_embed_reshaped = flat_embed.reshape(-1, self.pigment_embedding_dim)

            flat_absorption = self.absorption_coef_pred(flat_embed_reshaped)
            flat_scattering = self.scattering_coef_pred(flat_embed_reshaped)
            flat_reflectance = self.reflectance_pred(flat_embed_reshaped)
            flat_surface = self.surface_props_pred(flat_embed_reshaped)
            flat_fluorescence = self.fluorescence_pred(flat_embed_reshaped)

            absorption_coeffs = flat_absorption.reshape(batch_size, n_pigments, -1)
            scattering_coeffs = flat_scattering.reshape(batch_size, n_pigments, -1)
            spectral_reflectance = flat_reflectance.reshape(batch_size, n_pigments, -1)
            surface_props = flat_surface.reshape(batch_size, n_pigments, -1)
            fluorescence_props = flat_fluorescence.reshape(batch_size, n_pigments, -1)

            mixed_spectral = self._kubelka_munk_mixing(
                absorption_coeffs, scattering_coeffs, mixing_ratios
            )

            if mixing_ratios is None:
                mixing_ratios = torch.ones(batch_size, n_pigments, device=pigment_indices.device)
                mixing_ratios = mixing_ratios / n_pigments

            norm_ratios = mixing_ratios / (mixing_ratios.sum(dim=1, keepdim=True) + 1e-8)
            weighted_surface = surface_props * norm_ratios.unsqueeze(-1)
            mixed_surface = weighted_surface.sum(dim=1)

            if mixed_surface.size(1) < 5:
                padding = torch.zeros(batch_size, 5 - mixed_surface.size(1), 
                                device=mixed_surface.device)
                mixed_surface = torch.cat([mixed_surface, padding], dim=1)

            weighted_fluorescence = fluorescence_props * norm_ratios.unsqueeze(-1)
            mixed_fluorescence = weighted_fluorescence.sum(dim=1)
            weighted_embed = pigment_embed * norm_ratios.unsqueeze(-1)
            mixed_embed = weighted_embed.sum(dim=1)
        else:
            pigment_embed = self.pigment_embedding(pigment_indices)
    
            absorption_coeffs = self.absorption_coef_pred(pigment_embed)
            scattering_coeffs = self.scattering_coef_pred(pigment_embed)
            spectral_reflectance = self.reflectance_pred(pigment_embed)
            surface_props = self.surface_props_pred(pigment_embed)
            fluorescence_props = self.fluorescence_pred(pigment_embed)

            mixed_spectral = spectral_reflectance
            mixed_surface = surface_props
            mixed_fluorescence = fluorescence_props

            if mixed_surface.size(1) < 5:
                padding = torch.zeros(batch_size, 5 - mixed_surface.size(1), 
                                device=mixed_surface.device)
                mixed_surface = torch.cat([mixed_surface, padding], dim=1)
            
            mixed_embed = pigment_embed
            absorption_coeffs = absorption_coeffs.unsqueeze(1)
            scattering_coeffs = scattering_coeffs.unsqueeze(1)
            spectral_reflectance = spectral_reflectance.unsqueeze(1)
        
        # 1. Particle size effects
        if particle_data is not None:
            mixed_spectral = self._apply_particle_size_effects_vectorized(
                self.wavelengths,
                mixed_spectral, 
                particle_data
            )
            # 2. Apply aging if data provided
            if precomputed_aging is not None:
                if isinstance(precomputed_aging, np.ndarray):
                    precomputed_aging = torch.tensor(
                        precomputed_aging, 
                        device=mixed_spectral.device, 
                        dtype=mixed_spectral.dtype
                    )
                mixed_spectral = precomputed_aging
            
            # 3. Apply fluorescence effects
            if fluorescence_params is None:
                fluorescence_params = {
                    'emission_strength': mixed_fluorescence[:, 0],
                    'excitation_peak': mixed_fluorescence[:, 1],
                    'emission_peak': mixed_fluorescence[:, 2],
                    'bandwidth': mixed_fluorescence[:, 3]
                }
            for b in range(batch_size):
                if fluorescence_params['emission_strength'][b] > 0.05:
                    mixed_spectral[b] = self._simulate_fluorescence(
                        mixed_spectral[b],
                        {k: v[b].item() for k, v in fluorescence_params.items()}
                    )
            
            # 4. Handle layered rendering if configuration provided
            if layered_configuration is not None:
                layer_props = []
                
                for layer in layered_configuration:
                    indices = layer.get('pigment_indices')
                    ratios = layer.get('mixing_ratios')
                    thickness = layer.get('thickness', 20.0)
                    interface = layer.get('interface', {})
                    layer_result = self(indices, ratios)
                    refractive_index = None
                    if indices is not None and len(indices) > 0:
                        primary_idx = indices[0].item() if isinstance(indices, torch.Tensor) else indices[0]
                        if primary_idx < len(self.pigment_db.pigments):
                            pigment = self.pigment_db.pigments[primary_idx]
                            if hasattr(self.pigment_db, 'get_refractive_index'):
                                refractive_index = self.pigment_db.get_refractive_index(primary_idx)
                    
                    layer_props.append({
                        'spectral': layer_result['spectral'],
                        'thickness': thickness,
                        'interface': interface,
                        'refractive_index': refractive_index
                    })

                layered_spectral = self._calculate_layered_reflectance(layer_props)
                mixed_spectral = layered_spectral

        if torch.is_tensor(mixed_spectral) and mixed_spectral.dim() == 1:
            mixed_spectral = mixed_spectral.unsqueeze(0)  # Add batch dimension

        rgb_values = self._spectral_to_rgb_conversion(mixed_spectral)
        brdf_params = self._apply_brdf_parameters(rgb_values, mixed_surface)
        optical_params = {}
        if microstructure_params is not None:
            for key, value in microstructure_params.items():
                optical_params[key] = value

            if 'rms_height' in microstructure_params:
                scale_factor = 0.5 + microstructure_params['rms_height'] * 1.5
                if isinstance(brdf_params['roughness'], torch.Tensor):
                    brdf_params['roughness'] = torch.clamp(
                        brdf_params['roughness'] * scale_factor, 0.01, 1.0
                    )
                else:
                    brdf_params['roughness'] = min(1.0, brdf_params['roughness'] * scale_factor)

            if 'anisotropy' in microstructure_params:
                brdf_params['anisotropy'] = torch.tensor(
                    microstructure_params['anisotropy'],
                    device=brdf_params['anisotropy'].device,
                    dtype=brdf_params['anisotropy'].dtype
                )

        result = {
            'rgb': rgb_values,
            'spectral': mixed_spectral,
            'surface_props': mixed_surface,
            'brdf_params': brdf_params,
            'optical_params': optical_params,
            'fluorescence': fluorescence_params
        }

        if is_multi_pigment:
            result['pigment_embed'] = pigment_embed
            result['absorption_coeffs'] = absorption_coeffs
            result['scattering_coeffs'] = scattering_coeffs
                
        return result
    
    def simulate_full_rendering(self, 
                       pigment_indices, 
                       mixing_ratios=None, 
                       age_years=0, 
                       viewing_angle=0.0, 
                       light_direction=None,
                       env_conditions=None,
                       thermodynamic_validator=None):  
        """
        Perform full physically-based rendering simulation.
        
        Args:
            pigment_indices: Indices of pigments [batch_size, n_pigments]
            mixing_ratios: Mixing ratios for pigments
            age_years: Aging simulation in years
            viewing_angle: Viewing angle in radians
            light_direction: Light direction vector
            env_conditions: Environmental conditions dictionary
            thermodynamic_validator: Optional thermodynamic validator for aging
            
        Returns:
            Comprehensive rendering results with all physical effects
        """
        batch_size = pigment_indices.shape[0]
        device = pigment_indices.device
        if light_direction is None:
            light_direction = torch.tensor([0.0, 0.0, 1.0], device=device)
        if env_conditions is None:
            env_conditions = {
                'humidity': 0.5,
                'light_exposure': 0.5,
                'air_pollutants': 0.1
            }
        
        # 1. Calculate basic optical properties
        base_result = self(pigment_indices, mixing_ratios)
        
        # 2. Calculate aged reflectance if needed
        if age_years > 0:
            if thermodynamic_validator is not None:
                if len(pigment_indices.shape) > 1:
                    pigment_list = [
                        {
                            'indices': pigment_indices[b],
                            'ratios': mixing_ratios[b] if mixing_ratios is not None else None
                        }
                        for b in range(batch_size)
                    ]
                else:
                    pigment_list = [
                        {
                            'indices': pigment_indices[b:b+1],
                            'ratios': None
                        }
                        for b in range(batch_size)
                    ]
                aged_spectral_list = thermodynamic_validator.simulate_aging(
                    pigment_list, 
                    age_years=age_years,
                    conditions=env_conditions
                )
                if aged_spectral_list is not None and len(aged_spectral_list) == batch_size:
                    aged_spectral = torch.stack(aged_spectral_list, dim=0)
                    aged_rgb = self._spectral_to_rgb_conversion(aged_spectral)
                    base_result['rgb'] = aged_rgb
                    base_result['spectral'] = aged_spectral
                    base_result['aged'] = True
                    base_result['age_years'] = age_years
                    base_result['aging_method'] = 'thermodynamic_validator'
            else:
                aged_spectral = base_result['spectral'].clone()
                aging_params = {
                    'white': {'darkening': 0.05, 'yellowing': 0.1, 'wavelength_dependency': 0.3},
                    'red': {'darkening': 0.4, 'yellowing': 0.2, 'wavelength_dependency': 0.7},
                    'blue': {'darkening': 0.3, 'yellowing': 0.05, 'wavelength_dependency': 0.5,
                        'green_shift': 0.2},
                    'organic': {'darkening': -0.1, 'yellowing': 0.3, 'wavelength_dependency': 0.9,
                            'fading': 0.5},
                    'default': {'darkening': 0.2, 'yellowing': 0.15, 'wavelength_dependency': 0.5}
                }
                
                for b in range(batch_size):
                    if len(pigment_indices.shape) > 1:
                        sample_pigments = pigment_indices[b]
                    else:
                        sample_pigments = pigment_indices[b:b+1]
                    pigment_types = []
                    for p in sample_pigments:
                        p_id = p.item()
                        if p_id in [0, 1, 2, 3, 4, 5]:      # White pigments
                            pigment_types.append('white')
                        elif p_id in [10, 11, 14]:          # Red/mercury pigments
                            pigment_types.append('red')
                        elif p_id in [17, 18, 19]:          # Blue/copper pigments
                            pigment_types.append('blue')
                        elif p_id in [30, 31, 32, 33]:      # Organic pigments
                            pigment_types.append('organic')
                        else:
                            pigment_types.append('default')

                    if not pigment_types:
                        pigment_type = 'default'
                    else:
                        from collections import Counter
                        pigment_type = Counter(pigment_types).most_common(1)[0][0]

                    params = aging_params[pigment_type]
                    age_factor = age_years / 100.0 
                    humidity = env_conditions.get('humidity', 0.5)
                    light = env_conditions.get('light_exposure', 0.5)
                    env_modifier = 1.0 + (humidity - 0.5) * 0.5 + (light - 0.5) * 1.0
                    wavelengths = self.wavelengths.to(device)

                    darkening = params['darkening'] * age_factor * env_modifier
                    dark_factor = 1.0 - darkening * torch.exp(-(wavelengths - 450)**2 / 10000)

                    yellowing = params['yellowing'] * age_factor * env_modifier
                    yellow_factor = torch.ones_like(wavelengths)
                    yellow_factor = torch.where(wavelengths < 500, 
                                            1.0 - yellowing * (1.0 - (wavelengths - 400) / 100),
                                            yellow_factor)
                    yellow_factor = torch.where((wavelengths >= 550) & (wavelengths < 600),
                                            1.0 + yellowing * 0.2,
                                            yellow_factor)
                    
                    if 'green_shift' in params and params['green_shift'] > 0:
                        green_shift = params['green_shift'] * age_factor
                        green_factor = 1.0 + green_shift * torch.exp(-(wavelengths - 520)**2 / 1000)
                    else:
                        green_factor = torch.ones_like(wavelengths)
                    
                    if 'fading' in params and params['fading'] > 0:
                        fading = params['fading'] * age_factor * light
                        fading_factor = fading * torch.ones_like(wavelengths)
                        aged_spectral[b] = aged_spectral[b] * (1.0 - fading_factor) + fading_factor
                    aged_spectral[b] = aged_spectral[b] * dark_factor * yellow_factor * green_factor
                    aged_spectral[b] = torch.clamp(aged_spectral[b], 0.0, 1.0)

                aged_rgb = self._spectral_to_rgb_conversion(aged_spectral)
                base_result['rgb'] = aged_rgb
                base_result['spectral'] = aged_spectral
                base_result['aged'] = True
                base_result['age_years'] = age_years
                base_result['aging_method'] = 'enhanced_approximation'
        
        # 3. Calculate view-dependent BRDF effects
        if viewing_angle != 0.0:
            roughness = base_result['brdf_params']['roughness']
            specular = base_result['brdf_params']['specular']
            f0 = base_result['brdf_params']['f0']

            cos_theta = np.cos(viewing_angle)
            fresnel = f0 + (1 - f0) * ((1 - cos_theta) ** 5)
            alpha = roughness * roughness
            d_denom = (cos_theta * cos_theta * (alpha * alpha - 1) + 1)
            d_term = alpha * alpha / (np.pi * d_denom * d_denom)
            
            g_denom = cos_theta * (1 - alpha) + alpha
            g_term = (2 * cos_theta) / (cos_theta + torch.sqrt(alpha * alpha + (1 - alpha * alpha) * cos_theta * cos_theta))
            spec_contrib = fresnel * d_term * g_term * specular

            view_rgb = base_result['rgb'] + spec_contrib.unsqueeze(-1)
            view_rgb = torch.clamp(view_rgb, 0.0, 1.0)
            base_result['view_dependent_rgb'] = view_rgb
        
        return base_result
