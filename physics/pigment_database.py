"""
Pigment database module for Dunhuang murals, with physically accurate spectral properties.
This code serves as the foundation for the Physics Refinement Module.

@Author: Yuming Xie
"""
import torch
import numpy as np
from collections import defaultdict
import re
import warnings
import os
import json


def _parse_simple_formula(formula):
    """Parse a simple chemical formula without parentheses"""
    elements = {}
    pattern = r'([A-Z][a-z]*)(\d*)'
    matches = re.findall(pattern, formula)
        
    for element, count in matches:
        elements[element] = elements.get(element, 0) + (int(count) if count else 1)
        
    return elements

def _parse_single_compound(compound):
    """Parse a single chemical compound formula"""
    elements = {}
    if '(' in compound and ')' in compound:
        pattern = r'\(([^)]+)\)(\d*)'
        for group, multiplier in re.findall(pattern, compound):
            mult = int(multiplier) if multiplier else 1
            group_elements = _parse_simple_formula(group)
            for element, count in group_elements.items():
                elements[element] = elements.get(element, 0) + count * mult

        compound = re.sub(pattern, '', compound)
            
    simple_elements = _parse_simple_formula(compound)
    for element, count in simple_elements.items():
        elements[element] = elements.get(element, 0) + count
            
    return elements

def parse_formula(formula):
    """Chemical formula parser that handles hydrates, mixtures, and complex structures"""
    result = {}
    components = formula.split("+")
        
    for component in components:
        component = component.strip()
        if component.lower() in ('various', 'mixed', 'clay', 'amorphous'):
            continue
        if "·" in component or "." in component:
            if "·" in component:
                parts = component.split("·")
            else:
                parts = component.split(".")
                    
            base_formula = parts[0].strip()
            hydrate = parts[1].strip()

            base_elements = _parse_single_compound(base_formula)
            for element, count in base_elements.items():
                result[element] = result.get(element, 0) + count

            if hydrate.endswith("H2O"):
                water_count = 1
                if len(hydrate) > 3:
                    try:
                        water_count = int(hydrate[0:-3])
                    except ValueError:
                        water_count = 1

                result['H'] = result.get('H', 0) + 2 * water_count
                result['O'] = result.get('O', 0) + water_count
            else:
                hydrate_elements = _parse_single_compound(hydrate)
                for element, count in hydrate_elements.items():
                    result[element] = result.get(element, 0) + count
        else:
            elements = _parse_single_compound(component)
            for element, count in elements.items():
                result[element] = result.get(element, 0) + count
                    
    return result


# ===== Dunhuang Pigment Database =====
class DunhuangPigmentDB:
    """
    Database of historical pigments used in Dunhuang murals with physically accurate
    spectral properties, chemical composition analysis, and validation capabilities.
    
    Paper Link: https://skytyz.dha.ac.cn/CN/Y2022/V1/I1/47 
    
    Provides structured access to pigment properties including:
    - Full spectral reflectance data
    - Chemical composition and bond structures
    - Surface optical properties including roughness and microstructure
    - Aging characteristics
    - Historical usage patterns
    - Particle size distributions
    """
    def __init__(self, spectral_resolution=31, data_path=None, use_cache=True):
        """
        Initialize the pigment database with physical properties.
        
        Args:
            spectral_resolution: Number of spectral samples from 400-700nm
            data_path: Optional path to custom pigment data
            use_cache: Whether to cache computed values for performance
        """
        self.spectral_resolution = spectral_resolution
        self.wavelengths = np.linspace(400, 700, spectral_resolution)
        self.data_path = data_path
        self.use_cache = use_cache
        self._cache = {}
        self.pigments = self._init_pigment_data()
        self._validate_physical_properties()
        self.color_to_pigment = self._init_color_mapping()
        self.pigment_to_chemical = self._init_chemical_mapping()
        self.id_to_name = {i: p["name"] for i, p in enumerate(self.pigments)}
        self.name_to_id = {p["name"]: i for i, p in enumerate(self.pigments)}
        self.refractive_indices = self._init_refractive_indices()
        
    def _validate_physical_properties(self):
        """Validate physical properties to ensure they are within realistic bounds."""
        for i, pigment in enumerate(self.pigments):
            name = pigment['name']

            if "spectral_reflectance" in pigment:
                values = pigment["spectral_reflectance"]["values"]
                if not all(0 <= r <= 1 for r in values):
                    invalid_values = [r for r in values if r < 0 or r > 1]
                    warnings.warn(f"Invalid spectral reflectance values for {name}: {invalid_values}")
                    pigment["spectral_reflectance"]["values"] = [max(0, min(1, r)) for r in values]
            
            if "reflectance" in pigment:
                if not all(0 <= r <= 1 for r in pigment["reflectance"]):
                    warnings.warn(f"Invalid RGB reflectance for {name}: {pigment['reflectance']}")
                    pigment["reflectance"] = [max(0, min(1, r)) for r in pigment["reflectance"]]

            if not 0 <= pigment["roughness"] <= 1:
                warnings.warn(f"Invalid roughness for {name}: {pigment['roughness']}")
                pigment["roughness"] = max(0, min(1, pigment["roughness"]))

            for factor, value_dict in pigment["aging"].items():
                if isinstance(value_dict, dict):
                    for param, param_value in value_dict.items():
                        if not 0 <= param_value <= 1:
                            warnings.warn(f"Invalid aging factor {factor}.{param} for {name}: {param_value}")
                            value_dict[param] = max(0, min(1, param_value))
                else:
                    if not 0 <= value_dict <= 1:
                        warnings.warn(f"Invalid aging factor {factor} for {name}: {value_dict}")
                        pigment["aging"][factor] = max(0, min(1, value_dict))

            if pigment["particle_size"]["mean"] <= 0:
                warnings.warn(f"Invalid particle size mean for {name}: {pigment['particle_size']['mean']}")
                pigment["particle_size"]["mean"] = max(0.1, pigment["particle_size"]["mean"])
                
            if pigment["particle_size"]["std"] <= 0:
                warnings.warn(f"Invalid particle size std dev for {name}: {pigment['particle_size']['std']}")
                pigment["particle_size"]["std"] = max(0.01, pigment["particle_size"]["std"])
    
    def _init_pigment_data(self):
        """Initialize pigment data with spectral and physical properties."""
        if self.data_path and os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    custom_pigments = json.load(f)
                    print(f"Loaded {len(custom_pigments)} pigments from {self.data_path}")
                    return custom_pigments
            except Exception as e:
                warnings.warn(f"Failed to load custom pigment data: {e}. Using default data.")
        
        basic_pigments = self._init_basic_pigment_data()
        enhanced_pigments = []
        
        for pigment in basic_pigments:
            enhanced_pigment = pigment.copy()
            rgb = pigment["reflectance"]
            enhanced_pigment["spectral_reflectance"] = {
                "wavelengths": list(self.wavelengths),
                "values": self._rgb_to_spectral(rgb)
            }

            if "Cu" in self._parse_chemical_formula(pigment["formula"]):
                enhanced_pigment["refractive_index"] = 1.7  # Copper-based pigments
            elif "Pb" in self._parse_chemical_formula(pigment["formula"]):
                enhanced_pigment["refractive_index"] = 2.0  # Lead-based pigments
            elif "Hg" in self._parse_chemical_formula(pigment["formula"]):
                enhanced_pigment["refractive_index"] = 2.8  # Mercury-based pigments
            elif any(c in pigment["formula"] for c in ["C", "H", "N", "O"]) and len(pigment["formula"]) > 5:
                enhanced_pigment["refractive_index"] = 1.5  # Organic pigments
            else:
                enhanced_pigment["refractive_index"] = 1.6  # Default value

            basic_aging = pigment["aging"]
            enhanced_pigment["aging"] = {
                "yellowing": {"rate": basic_aging["yellowing"] / 20, "max": basic_aging["yellowing"]},
                "darkening": {"rate": basic_aging["darkening"] / 20, "max": basic_aging["darkening"]},
                "fading": {"rate": basic_aging["fading"] / 20, "max": basic_aging["fading"]}
            }

            enhanced_pigment["microstructure"] = {
                "porosity": 0.2 + 0.3 * pigment["roughness"],  # Correlate with roughness
                "specific_surface_area": 5.0 + 15.0 * (1.0 / pigment["particle_size"]["mean"]),  # m²/g, inverse with size
                "shape_factor": 0.7 if pigment["color"] in ["red", "yellow"] else 0.5  # More spherical for some colors
            }

            enhanced_pigment["chemical_stability"] = self._get_chemical_stability(pigment)
            enhanced_pigment["spectral_properties"] = self._calculate_spectral_properties(enhanced_pigment)
            
            enhanced_pigments.append(enhanced_pigment)
            
        return enhanced_pigments
    
    def _calculate_spectral_properties(self, pigment):
        """Calculate absorption and scattering coefficients for Kubelka-Munk theory."""
        properties = {}
        if "spectral_reflectance" in pigment:
            reflectance = np.array(pigment["spectral_reflectance"]["values"])
        else:
            reflectance = np.array(self._rgb_to_spectral(pigment["reflectance"]))
        
        # Calculate K/S ratio using Kubelka-Munk equation
        # (1-R)^2 / (2*R) = K/S
        reflectance = np.clip(reflectance, 0.001, 0.999)
        k_over_s = (1 - reflectance)**2 / (2 * reflectance)
        
        # Estimate K and S separately based on pigment type
        s_factor = 1.0 
    
        # Adjust scattering based on particle size
        if "particle_size" in pigment:
            size = pigment["particle_size"]["mean"]
            if size < 1.0:
                s_factor = 2.0
            elif size < 3.0:
                s_factor = 1.5
            elif size > 8.0:
                s_factor = 0.7
        
        # Adjust scattering based on microstructure
        if "microstructure" in pigment and "porosity" in pigment["microstructure"]:
            porosity = pigment["microstructure"]["porosity"]
            s_factor *= (1.0 + porosity)
        
        # Calculate S and K coefficients
        s_values = np.ones_like(reflectance) * s_factor
        k_values = k_over_s * s_values
        
        properties["k_values"] = k_values.tolist()
        properties["s_values"] = s_values.tolist()
        properties["k_over_s"] = k_over_s.tolist()
        
        return properties
    
    def _init_basic_pigment_data(self):
        """Pigment data from research paper (unchanged from original)"""
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
            
            {"name": "quartz", "color": "white", "formula": "SiO2", 
            "reflectance": [0.91, 0.91, 0.91], "roughness": 0.28,
            "aging": {"yellowing": 0.05, "darkening": 0.08, "fading": 0.05},
            "particle_size": {"mean": 6.0, "std": 1.3, "unit": "micron"}},
            
            {"name": "lead_chloride", "color": "white", "formula": "PbCl2", 
            "reflectance": [0.94, 0.94, 0.94], "roughness": 0.22,
            "aging": {"yellowing": 0.18, "darkening": 0.5, "fading": 0.1},
            "particle_size": {"mean": 3.0, "std": 0.7, "unit": "micron"}},
            
            {"name": "muscovite", "color": "white", "formula": "KAl2(Si3Al)O10(OH)2", 
            "reflectance": [0.89, 0.89, 0.89], "roughness": 0.32,
            "aging": {"yellowing": 0.12, "darkening": 0.15, "fading": 0.1},
            "particle_size": {"mean": 7.5, "std": 1.8, "unit": "micron"}},
            
            {"name": "talc", "color": "white", "formula": "Mg3Si4O10(OH)2", 
            "reflectance": [0.88, 0.88, 0.88], "roughness": 0.35,
            "aging": {"yellowing": 0.13, "darkening": 0.14, "fading": 0.12},
            "particle_size": {"mean": 4.5, "std": 1.1, "unit": "micron"}},
            
            {"name": "anhydrite", "color": "white", "formula": "CaSO4", 
            "reflectance": [0.9, 0.9, 0.9], "roughness": 0.26,
            "aging": {"yellowing": 0.18, "darkening": 0.15, "fading": 0.08},
            "particle_size": {"mean": 7.0, "std": 1.4, "unit": "micron"}},
            
            {"name": "lead_sulfate", "color": "white", "formula": "PbSO4", 
            "reflectance": [0.92, 0.92, 0.92], "roughness": 0.25,
            "aging": {"yellowing": 0.17, "darkening": 0.5, "fading": 0.08},
            "particle_size": {"mean": 3.2, "std": 0.9, "unit": "micron"}},
            
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
            
            {"name": "realgar", "color": "red", "formula": "As4S4", 
            "reflectance": [0.85, 0.6, 0.05], "roughness": 0.4,
            "aging": {"yellowing": 0.1, "darkening": 0.3, "fading": 0.5},
            "particle_size": {"mean": 2.5, "std": 0.7, "unit": "micron"}},
            
            {"name": "lead_oxide", "color": "red", "formula": "Pb3O4", 
            "reflectance": [0.85, 0.25, 0.1], "roughness": 0.3,
            "aging": {"yellowing": 0.1, "darkening": 0.7, "fading": 0.2},
            "particle_size": {"mean": 3.0, "std": 0.7, "unit": "micron"}},
            
            {"name": "cinnabar_variant", "color": "red", "formula": "HgS", 
            "reflectance": [0.83, 0.09, 0.09], "roughness": 0.42,
            "aging": {"yellowing": 0.06, "darkening": 0.22, "fading": 0.28},
            "particle_size": {"mean": 1.7, "std": 0.55, "unit": "micron"}},
            
            {"name": "red_ochre", "color": "red", "formula": "Fe2O3 + SiO2 + clay", 
            "reflectance": [0.75, 0.2, 0.18], "roughness": 0.48,
            "aging": {"yellowing": 0.12, "darkening": 0.45, "fading": 0.15},
            "particle_size": {"mean": 4.5, "std": 1.2, "unit": "micron"}},
            
            # Blue pigments (Table 3)
            {"name": "azurite", "color": "blue", "formula": "Cu3(CO3)2(OH)2", 
            "reflectance": [0.1, 0.3, 0.8], "roughness": 0.45,
            "aging": {"yellowing": 0.3, "darkening": 0.4, "fading": 0.2},
            "particle_size": {"mean": 7.0, "std": 1.8, "unit": "micron"}},
            
            {"name": "lapis_lazuli", "color": "blue", "formula": "Na3Ca(Al3Si3O12)S", 
            "reflectance": [0.15, 0.25, 0.85], "roughness": 0.4,
            "aging": {"yellowing": 0.1, "darkening": 0.2, "fading": 0.15},
            "particle_size": {"mean": 10.0, "std": 2.5, "unit": "micron"}},
            
            {"name": "synthetic_blue", "color": "blue", "formula": "Na6Al4Si6S4O20", 
            "reflectance": [0.2, 0.4, 0.82], "roughness": 0.35,
            "aging": {"yellowing": 0.2, "darkening": 0.3, "fading": 0.25},
            "particle_size": {"mean": 5.0, "std": 1.0, "unit": "micron"}},
            
            # Green pigments (Table 4)
            {"name": "atacamite", "color": "green", "formula": "Cu2Cl(OH)3", 
            "reflectance": [0.1, 0.75, 0.2], "roughness": 0.4,
            "aging": {"yellowing": 0.25, "darkening": 0.35, "fading": 0.2},
            "particle_size": {"mean": 6.0, "std": 1.2, "unit": "micron"}},
            
            {"name": "malachite", "color": "green", "formula": "Cu2(CO3)(OH)2", 
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
            
            {"name": "arsenic_sulfide", "color": "yellow", "formula": "As2S3", 
            "reflectance": [0.88, 0.78, 0.08], "roughness": 0.36,
            "aging": {"yellowing": 0.12, "darkening": 0.22, "fading": 0.38},
            "particle_size": {"mean": 2.2, "std": 0.65, "unit": "micron"}},
            
            # Black pigments
            {"name": "carbon_black", "color": "black", "formula": "C (amorphous)", 
            "reflectance": [0.05, 0.05, 0.05], "roughness": 0.6,
            "aging": {"yellowing": 0.0, "darkening": 0.05, "fading": 0.1},
            "particle_size": {"mean": 1.0, "std": 0.4, "unit": "micron"}},
            
            # Brown pigments (alteration products)
            {"name": "brown_alteration", "color": "brown", "formula": "various", 
            "reflectance": [0.4, 0.25, 0.15], "roughness": 0.55,
            "aging": {"yellowing": 0.15, "darkening": 0.3, "fading": 0.2},
            "particle_size": {"mean": 3.0, "std": 1.0, "unit": "micron"}},
            
            # Gray pigments (mixtures)
            {"name": "gray_mixture", "color": "gray", "formula": "mixed", 
            "reflectance": [0.5, 0.5, 0.5], "roughness": 0.4,
            "aging": {"yellowing": 0.2, "darkening": 0.25, "fading": 0.15},
            "particle_size": {"mean": 4.0, "std": 1.5, "unit": "micron"}},
            
            # Gold
            {"name": "gold_leaf", "color": "gold", "formula": "Au", 
            "reflectance": [0.85, 0.7, 0.1], "roughness": 0.1,
            "aging": {"yellowing": 0.0, "darkening": 0.05, "fading": 0.02},
            "particle_size": {"mean": 0.1, "std": 0.02, "unit": "micron"}},
            
            # Organic pigments
            {"name": "indigo", "color": "blue", "formula": "C16H10N2O2", 
            "reflectance": [0.1, 0.2, 0.6], "roughness": 0.45,
            "aging": {"yellowing": 0.2, "darkening": 0.3, "fading": 0.5},
            "particle_size": {"mean": 2.0, "std": 0.8, "unit": "micron"}},
            
            {"name": "gamboge", "color": "yellow", "formula": "C38H44O8", 
            "reflectance": [0.8, 0.75, 0.1], "roughness": 0.4,
            "aging": {"yellowing": 0.15, "darkening": 0.25, "fading": 0.6},
            "particle_size": {"mean": 1.8, "std": 0.7, "unit": "micron"}},
            
            {"name": "lac", "color": "red", "formula": "C15H12O8", 
            "reflectance": [0.8, 0.15, 0.2], "roughness": 0.38,
            "aging": {"yellowing": 0.1, "darkening": 0.2, "fading": 0.7},
            "particle_size": {"mean": 1.5, "std": 0.6, "unit": "micron"}},
            
            {"name": "phellodendron", "color": "yellow", "formula": "C20H18O4", 
            "reflectance": [0.85, 0.8, 0.15], "roughness": 0.42,
            "aging": {"yellowing": 0.2, "darkening": 0.2, "fading": 0.65},
            "particle_size": {"mean": 1.7, "std": 0.65, "unit": "micron"}}
        ]
    
    def _rgb_to_spectral(self, rgb):
        """
        Convert RGB values to full spectral reflectance values with improved accuracy.
        
        Args:
            rgb: RGB reflectance values [r, g, b]
            
        Returns:
            List of spectral reflectance values across visible spectrum
        """
        if self.use_cache:
            cache_key = f"spectral_{rgb[0]:.4f}_{rgb[1]:.4f}_{rgb[2]:.4f}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        r, g, b = rgb
        spectral = np.zeros(self.spectral_resolution)
        wavelengths = self.wavelengths

        red_idx = (wavelengths >= 580)
        red_curve = np.zeros_like(spectral)
        if np.any(red_idx):
            x = wavelengths[red_idx]
            peak = 650
            left_width = 50
            right_width = 70
            left_mask = x < peak
            widths = np.zeros_like(x)
            widths[left_mask] = left_width
            widths[~left_mask] = right_width
            red_curve[red_idx] = r * np.exp(-((x - peak)**2) / (2 * widths**2))
        
        green_idx = (wavelengths >= 480) & (wavelengths <= 600)
        green_curve = np.zeros_like(spectral)
        if np.any(green_idx):
            x = wavelengths[green_idx]
            peak = 530
            width = 40
            green_curve[green_idx] = g * np.exp(-((x - peak)**2) / (2 * width**2))

        blue_idx = (wavelengths <= 520)
        blue_curve = np.zeros_like(spectral)
        if np.any(blue_idx):
            x = wavelengths[blue_idx]
            peak = 450
            left_width = 70
            right_width = 40

            left_mask = x < peak
            widths = np.zeros_like(x)
            widths[left_mask] = left_width
            widths[~left_mask] = right_width
            blue_curve[blue_idx] = b * np.exp(-((x - peak)**2) / (2 * widths**2))

        spectral = red_curve + green_curve + blue_curve
        gray_level = min(r, g, b)

        if gray_level > 0.1:  
            spectral += gray_level * 0.8 

        if r > 0.5 and g > 0.5 and b < 0.5:
            yellow_idx = (wavelengths >= 560) & (wavelengths <= 590)
            if np.any(yellow_idx):
                x = wavelengths[yellow_idx]
                peak = 575
                width = 15
                yellow_factor = min(r, g) * 0.5
                spectral[yellow_idx] += yellow_factor * np.exp(-((x - peak)**2) / (2 * width**2))

        if r > 0.3 and g < 0.5 and b < 0.3 and r > g > b:
            brown_idx = (wavelengths >= 600)
            if np.any(brown_idx):
                spectral[brown_idx] *= 1.2  
                spectral[wavelengths < 500] *= 0.8  
                
        spectral = np.clip(spectral, 0, 1)
        if self.use_cache:
            self._cache[cache_key] = spectral.tolist()
        
        return spectral.tolist()
    
    def _get_chemical_stability(self, pigment):
        """Determine chemical stability properties based on composition"""
        formula = pigment["formula"]
        elements = self._parse_chemical_formula(formula)
        stability = {
            "acid_resistance": 0.7, 
            "alkali_resistance": 0.7,
            "light_stability": 0.7,
            "thermal_stability": 0.7,
            "compatibility_class": "standard"
        }

        # Lead compounds - vulnerable to acids, sulfides
        if "Pb" in elements:
            stability["acid_resistance"] = 0.3
            stability["alkali_resistance"] = 0.8
            stability["thermal_stability"] = 0.8
            stability["compatibility_class"] = "lead-based"
        
        # Mercury compounds - volatile, light sensitive
        if "Hg" in elements:
            stability["thermal_stability"] = 0.4
            stability["light_stability"] = 0.5
            stability["compatibility_class"] = "mercury-based"
        
        # Copper compounds - react with acids, sulfides
        if "Cu" in elements:
            stability["acid_resistance"] = 0.4
            stability["alkali_resistance"] = 0.6
            stability["compatibility_class"] = "copper-based"
        
        # Arsenic compounds - light sensitive, toxic
        if "As" in elements:
            stability["light_stability"] = 0.3
            stability["compatibility_class"] = "arsenic-based"
        
        # Carbonates - react with acids
        if formula.lower().find("co3") >= 0:
            stability["acid_resistance"] = min(stability["acid_resistance"], 0.3)
        
        # Sulfides - oxidize in light/air
        if formula.lower().find("s") >= 0 and not formula.lower().find("so4") >= 0:
            stability["light_stability"] = min(stability["light_stability"], 0.5)
        
        # Organic compounds - typically less stable
        if any(e in ["C", "H", "N", "O"] for e in elements) and len(elements) > 3:
            stability["light_stability"] = min(stability["light_stability"], 0.4)
            stability["thermal_stability"] = min(stability["thermal_stability"], 0.5)
            stability["compatibility_class"] = "organic"
        
        # Clays and silicates - generally stable
        if any(e in ["Si", "Al"] for e in elements) and "O" in elements:
            stability["acid_resistance"] = max(stability["acid_resistance"], 0.7)
            stability["alkali_resistance"] = max(stability["alkali_resistance"], 0.7)
            stability["compatibility_class"] = "silicate"
        
        # Gold - extremely stable
        if "Au" in elements:
            stability["acid_resistance"] = 0.9
            stability["alkali_resistance"] = 0.9
            stability["light_stability"] = 0.9
            stability["thermal_stability"] = 0.9
            stability["compatibility_class"] = "noble-metal"
        
        return stability

    def _init_color_mapping(self):
        """Map colors to pigment indices based on paper data."""
        color_map = defaultdict(list)
        for i, pigment in enumerate(self.pigments):
            color_map[pigment["color"]].append(i)
        return dict(color_map)
    
    def _init_chemical_mapping(self):
        """Map pigments to chemical compositions for validation."""
        chem_map = {}
        for i, pigment in enumerate(self.pigments):
            chem_map[i] = self._parse_chemical_formula(pigment["formula"])
        return chem_map
    
    def _init_refractive_indices(self):
        """Initialize refractive indices for all pigments."""
        return {i: p.get("refractive_index", 1.5) for i, p in enumerate(self.pigments)}
    
    def _parse_chemical_formula(self, formula):
        """
        Parse chemical formula to element counts.
        
        Args:
            formula: Chemical formula string
            
        Returns:
            Dictionary of element counts
        """
        if formula.lower() in ('various', 'mixed', 'amorphous'):
            return {}
        
        try:
            return parse_formula(formula)
        except Exception as e:
            warnings.warn(f"Error parsing formula {formula}: {e} - using fallback parser")
            elements = defaultdict(int)
            for element in ["H", "C", "O", "N", "S", "Cu", "Fe", "Hg", "Pb", "As", 
                            "Ca", "Al", "Si", "Na", "K", "Mg", "Au", "Cl"]:
                if element in formula:
                    elements[element] = formula.count(element)
            
            return dict(elements)
    
    def get_spectral_reflectance(self, pigment_id):
        """
        Get full spectral reflectance data for a pigment.
        
        Args:
            pigment_id: Pigment identifier
            
        Returns:
            Tuple of (wavelengths, reflectance_values)
        """
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")

        if self.use_cache:
            cache_key = f"reflectance_{pigment_id}"
            if cache_key in self._cache:
                return self._cache[cache_key]
            
        pigment = self.pigments[pigment_id]
        if "spectral_reflectance" in pigment:
            result = (
                pigment["spectral_reflectance"]["wavelengths"],
                pigment["spectral_reflectance"]["values"]
            )
        else:
            result = (
                list(self.wavelengths),
                self._rgb_to_spectral(pigment["reflectance"])
            )

        if self.use_cache:
            self._cache[cache_key] = result
            
        return result
    
    def get_reflectance(self, pigment_id):
        """Get RGB reflectance for a pigment by ID."""
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        return torch.tensor(self.pigments[pigment_id]["reflectance"])
    
    def get_roughness(self, pigment_id):
        """Get roughness for a pigment by ID."""
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        return self.pigments[pigment_id]["roughness"]
    
    def get_aging_factors(self, pigment_id):
        """Get aging factors for a pigment by ID."""
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        return self.pigments[pigment_id]["aging"]
    
    def get_particle_data(self, pigment_id):
        """Get detailed particle data including size and microstructure."""
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        pigment = self.pigments[pigment_id]
        return {
            "size": pigment["particle_size"],
            "microstructure": pigment.get("microstructure", {
                "porosity": 0.2,
                "specific_surface_area": 5.0,
                "shape_factor": 0.5
            })
        }
    
    def get_chemical_properties(self, pigment_id):
        """Get chemical properties and stability data."""
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        pigment = self.pigments[pigment_id]
        return {
            "formula": pigment["formula"],
            "elements": self._parse_chemical_formula(pigment["formula"]),
            "stability": pigment.get("chemical_stability", {
                "acid_resistance": 0.7,
                "alkali_resistance": 0.7,
                "light_stability": 0.7,
                "thermal_stability": 0.7,
                "compatibility_class": "standard"
            }),
            "refractive_index": pigment.get("refractive_index", 1.5)
        }
    
    def get_pigments_by_color(self, color):
        """Get pigment IDs for a given color category."""
        if color not in self.color_to_pigment:
            warnings.warn(f"Unknown color category: {color}")
            return []
        return self.color_to_pigment.get(color, [])
    
    def get_chemical_signature(self, pigment_id):
        """Get chemical signature for validation."""
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        return self.pigment_to_chemical[pigment_id]
    
    def get_pigment_by_name(self, name):
        """Get pigment data by name."""
        if name not in self.name_to_id:
            for pigment_name in self.name_to_id:
                if name.lower() == pigment_name.lower():
                    name = pigment_name
                    break
            else:
                raise ValueError(f"Unknown pigment name: {name}")
            
        pigment_id = self.name_to_id[name]
        return self.pigments[pigment_id]
    
    def get_absorption_and_scattering(self, pigment_id):
        """
        Get absorption and scattering coefficients for Kubelka-Munk calculations.
        
        Args:
            pigment_id: Pigment identifier
            
        Returns:
            Dictionary with k_values and s_values
        """
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        pigment = self.pigments[pigment_id]
        if "spectral_properties" in pigment:
            return {
                "k_values": pigment["spectral_properties"]["k_values"],
                "s_values": pigment["spectral_properties"]["s_values"]
            }
        
        properties = self._calculate_spectral_properties(pigment)
        
        return {
            "k_values": properties["k_values"],
            "s_values": properties["s_values"]
        }
    
    def get_refractive_index(self, pigment_id):
        """
        Get refractive index data for a pigment.
        
        Args:
            pigment_id: Pigment identifier
            
        Returns:
            Refractive index value or dictionary with spectral data
        """
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        pigment = self.pigments[pigment_id]

        if "spectral_refractive_index" in pigment:
            return pigment["spectral_refractive_index"]
        
        return pigment.get("refractive_index", 1.5)
    
    def calculate_aged_reflectance(self, pigment_id, years, environmental_conditions=None):
        """
        Calculate aged spectral reflectance based on time-dependent aging model.
        
        Args:
            pigment_id: Pigment identifier
            years: Aging time in years
            environmental_conditions: Optional dictionary with environmental factors
                                     (humidity, light_exposure, pollutants)
            
        Returns:
            Aged spectral reflectance values
        """
        if self.use_cache:
            env_hash = 0
            if environmental_conditions:
                env_hash = hash(frozenset(environmental_conditions.items()))
            cache_key = f"aged_{pigment_id}_{years}_{env_hash}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        pigment = self.pigments[pigment_id]

        if "spectral_reflectance" in pigment:
            original_spectral = np.array(pigment["spectral_reflectance"]["values"])
        else:
            original_spectral = np.array(self._rgb_to_spectral(pigment["reflectance"]))

        if "aging" in pigment:
            aging = pigment["aging"]
            yellowing_rate = aging["yellowing"]["rate"]
            yellowing_max = aging["yellowing"]["max"]
            darkening_rate = aging["darkening"]["rate"]
            darkening_max = aging["darkening"]["max"]
            fading_rate = aging["fading"]["rate"]
            fading_max = aging["fading"]["max"]
        else:
            yellowing_rate, yellowing_max = 0.005, 0.1
            darkening_rate, darkening_max = 0.005, 0.1
            fading_rate, fading_max = 0.01, 0.2

        env_factor = 1.0
        if environmental_conditions:
            if "humidity" in environmental_conditions:
                humidity = environmental_conditions["humidity"]
                env_factor *= 1.0 + humidity * 0.5

            if "light_exposure" in environmental_conditions:
                light = environmental_conditions["light_exposure"]
                env_factor *= 1.0 + light * 0.8

            if "pollutants" in environmental_conditions:
                pollutants = environmental_conditions["pollutants"]
                env_factor *= 1.0 + pollutants * 0.7
        
        # Calculate time-dependent aging factors
        # f(t) = max_value * (1 - exp(-rate * t))
        yellowing = yellowing_max * (1 - np.exp(-yellowing_rate * years * env_factor))
        darkening = darkening_max * (1 - np.exp(-darkening_rate * years * env_factor))
        fading = fading_max * (1 - np.exp(-fading_rate * years * env_factor))
        
        yellowing_effect = np.ones_like(original_spectral)
        blue_region = self.wavelengths < 480
        if np.any(blue_region):
            yellowing_effect[blue_region] -= yellowing * (1.5 - self.wavelengths[blue_region] / 480)

        darkening_effect = np.ones_like(original_spectral) * (1.0 - darkening)
        fading_effect = np.ones_like(original_spectral)
        for i, refl in enumerate(original_spectral):
            if refl < 0.5:      # Absorption band
                fading_effect[i] = 1.0 + fading * (0.5 - refl) * 2.0
            else:               # Reflection band
                fading_effect[i] = 1.0
        
        aged_spectral = original_spectral * yellowing_effect * darkening_effect * fading_effect
        aged_spectral = np.clip(aged_spectral, 0.0, 1.0)
        if self.use_cache:
            self._cache[cache_key] = aged_spectral.tolist()
        
        return aged_spectral.tolist()
    
    def suggest_compatible_pigments(self, pigment_id):
        """
        Suggest pigments that are historically and chemically compatible.
        
        Args:
            pigment_id: Base pigment to find compatible options for
            
        Returns:
            List of compatible pigment IDs with compatibility scores
        """
        if pigment_id >= len(self.pigments):
            raise ValueError(f"Invalid pigment ID: {pigment_id}")
            
        base_pigment = self.pigments[pigment_id]
        base_chem = self._parse_chemical_formula(base_pigment["formula"])
        base_class = base_pigment.get("chemical_stability", {}).get("compatibility_class", "standard")

        if self.use_cache:
            cache_key = f"compatible_{pigment_id}"
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        compatibility_scores = []
        
        for i, candidate in enumerate(self.pigments):
            if i == pigment_id:
                continue  

            candidate_chem = self._parse_chemical_formula(candidate["formula"])
            candidate_class = candidate.get("chemical_stability", {}).get("compatibility_class", "standard")
            score = 1.0
            incompatible_pairs = [
                ("lead-based", "mercury-based", 0.3),
                ("lead-based", "arsenic-based", 0.4),
                ("copper-based", "mercury-based", 0.3),
                ("copper-based", "arsenic-based", 0.3),
                ("organic", "lead-based", 0.6),
                ("organic", "copper-based", 0.7)
            ]
            
            for class1, class2, penalty in incompatible_pairs:
                if (base_class == class1 and candidate_class == class2) or \
                   (base_class == class2 and candidate_class == class1):
                    score *= penalty

            problematic_combinations = [
                ({"Pb"}, {"S"}, 0.3),  # Lead + Sulfur
                ({"Cu"}, {"S"}, 0.4),  # Copper + Sulfur
                ({"Fe"}, {"Pb"}, 0.8)  # Iron + Lead
            ]
            
            for elements1, elements2, penalty in problematic_combinations:
                if any(e in base_chem for e in elements1) and any(e in candidate_chem for e in elements2):
                    score *= penalty

            if base_pigment["color"] == candidate["color"]:
                score *= 1.2
            
            base_size = base_pigment["particle_size"]["mean"]
            candidate_size = candidate["particle_size"]["mean"]

            size_ratio = min(base_size, candidate_size) / max(base_size, candidate_size)
            score *= 0.8 + 0.2 * size_ratio
            
            compatibility_scores.append({
                "pigment_id": i,
                "name": candidate["name"],
                "color": candidate["color"],
                "compatibility_score": min(1.0, score)
            })

        compatibility_scores.sort(key=lambda x: x["compatibility_score"], reverse=True)

        if self.use_cache:
            self._cache[cache_key] = compatibility_scores
        
        return compatibility_scores
    
    def get_spectral_data_tensor(self, pigment_id):
        """
        Get spectral data as a tensor.
        
        Args:
            pigment_id: Pigment identifier
            
        Returns:
            Tensor of spectral reflectance data
        """
        _, reflectance = self.get_spectral_reflectance(pigment_id)
        return torch.tensor(reflectance, dtype=torch.float32)
    
    def export_to_json(self, file_path):
        """
        Export pigment database to JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.pigments, f, indent=2)
            print(f"Successfully exported database to {file_path}")
        except Exception as e:
            warnings.warn(f"Failed to export database: {e}")
    
    def clear_cache(self):
        """Clear the computation cache."""
        self._cache = {}
        print("Cache cleared")
    
    def get_cache_stats(self):
        """Get statistics about the cache."""
        if not self.use_cache:
            return {"enabled": False, "size": 0}
        
        return {
            "enabled": True,
            "size": len(self._cache),
            "memory_estimate_kb": sum(len(str(v)) for v in self._cache.values()) / 1024
        }
    
def safe_compare(value, threshold, comparison='gt', aggregation='all'):
    """
    Safely compare values which might be scalars, numpy arrays, or tensors.
    
    Args:
        value: Value to compare (scalar, numpy array or tensor)
        threshold: Threshold value
        comparison: Type of comparison ('gt', 'lt', 'ge', 'le', 'eq')
        aggregation: How to aggregate array results ('all', 'any')
        
    Returns:
        Boolean result of comparison
    """
    if isinstance(value, (np.ndarray, torch.Tensor)):
        if comparison == 'gt':
            result = value > threshold
        elif comparison == 'lt':
            result = value < threshold
        elif comparison == 'ge':
            result = value >= threshold
        elif comparison == 'le':
            result = value <= threshold
        elif comparison == 'eq':
            result = value == threshold

        if isinstance(value, np.ndarray):
            return np.all(result) if aggregation == 'all' else np.any(result)
        else: 
            return torch.all(result).item() if aggregation == 'all' else torch.any(result).item()
    else:
        if comparison == 'gt':
            return value > threshold
        elif comparison == 'lt':
            return value < threshold
        elif comparison == 'ge':
            return value >= threshold
        elif comparison == 'le':
            return value <= threshold
        elif comparison == 'eq':
            return value == threshold