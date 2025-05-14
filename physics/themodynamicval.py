"""
Thermodynamic Validator for Dunhuang mural pigments.

This module provides a validator for pigment mixture stability based on chemical 
thermodynamics, reaction kinetics, spectral validation, non-linear aging models, 
quantum efficiency, and nanoscale physics for Dunhuang mural pigments.
"""
import torch
import numpy as np

from pigment_database import DunhuangPigmentDB, safe_compare

# ===== Thermodynamic Validator =====
class ThermodynamicValidator:
    """
    Validator for pigment mixture stability.
    """
    def __init__(self, pigment_db=None):
        self.pigment_db = pigment_db if pigment_db else DunhuangPigmentDB()
        self._create_pigment_mappings()
        self.reaction_parameters = self._init_reaction_parameters()
        self.quantum_efficiency = self._init_quantum_efficiency()
        self.nanoscale_properties = self._init_nanoscale_properties()
        self.microbial_susceptibility = self._init_microbial_susceptibility()
        self.ph_sensitivity = self._init_ph_sensitivity()
        self.historical_data = self._init_historical_data()
        self.binder_compatibility = self._init_binder_compatibility()
        self.spectral_data = self._init_spectral_data()
    
    def _element_in_list(self, element, target_list):
        """
        Check if a scalar or array element is in a list.
        
        Args:
            element: Value to check (scalar, numpy array or tensor)
            target_list: List of values to check against
            
        Returns:
            Boolean result
        """
        if isinstance(element, (np.ndarray, torch.Tensor)):
            if hasattr(element, 'numel') and element.numel() == 1:
                return element.item() in target_list
            elif hasattr(element, 'size') and element.size == 1:
                return element.item() in target_list
            else: 
                if isinstance(element, np.ndarray):
                    return any(e in target_list for e in element.flatten())
                else: 
                    return any(e.item() in target_list for e in element.flatten())
        else:
            return element in target_list
    
    def _preprocess_indices(self, indices):
        """
        Convert tensor indices to plain Python scalars.
        """
        if isinstance(indices, torch.Tensor) or isinstance(indices, np.ndarray):
            if len(indices.shape) > 1:
                return [[self._extract_scalar(item) for item in row] for row in indices]
            return [self._extract_scalar(item) for item in indices]
        elif isinstance(indices, list) and any(isinstance(item, (torch.Tensor, np.ndarray)) for item in indices):
            return [self._extract_scalar(item) for item in indices]

        return indices

    def _extract_scalar(self, value):
        """
        Extract scalar value from tensor, array, or list.
        """
        if isinstance(value, list):
            if len(value) == 1:
                return self._extract_scalar(value[0])
            elif len(value) > 1:
                return self._extract_scalar(value[0])
            else:
                return None
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            else:
                return value.flat[0].item() if value.size > 0 else None
        elif hasattr(value, 'item'):
            if hasattr(value, 'numel') and value.numel() > 1:
                return value[0].item()
            elif hasattr(value, 'size') and (isinstance(value.size, int) and value.size > 1 or 
                                        hasattr(value.size, '__len__') and len(value.size) > 0 and value.size[0] > 1):
                return value[0].item()
            else:
                return value.item()
        return value

    def _create_pigment_mappings(self):
        """Create pigment mappings for faster lookups."""
        self.id_to_chemical = {}
        self.name_to_id = {}
        
        for i, pigment in enumerate(self.pigment_db.pigments):
            if i < len(self.pigment_db.pigments):
                formula = pigment.get("formula", "")
                self.id_to_chemical[i] = formula
                self.name_to_id[pigment["name"]] = i

    def _init_reaction_parameters(self):
        """
        Initialize Arrhenius parameters for chemical reactions.
        
        Format: (pigment1_id, pigment2_id, activation_energy, pre_exponential_factor, reaction_type)
        Activation energy in kJ/mol, pre-exponential factor in appropriate units
        """
        return [
            # Lead-based pigments with sulfur-containing pigments
            (3, 10, 58.2, 5.4e3, "chemical_reaction"),  # Lead white with cinnabar
            (3, 11, 57.8, 5.2e3, "chemical_reaction"),  # Lead white with vermilion
            (3, 13, 62.5, 4.8e3, "chemical_reaction"),  # Lead white with realgar
            (5, 10, 60.1, 4.9e3, "chemical_reaction"),  # Lead chloride with cinnabar
            (5, 11, 59.8, 4.9e3, "chemical_reaction"),  # Lead chloride with vermilion
            (9, 10, 65.3, 3.8e3, "chemical_reaction"),  # Lead sulfate with cinnabar
            (9, 11, 65.0, 3.7e3, "chemical_reaction"),  # Lead sulfate with vermilion
            (14, 10, 61.2, 4.7e3, "chemical_reaction"), # Lead oxide with cinnabar
            (14, 11, 61.0, 4.5e3, "chemical_reaction"), # Lead oxide with vermilion
            
            # Copper-based pigments and sulfides
            (17, 10, 54.7, 6.2e3, "chemical_reaction"), # Azurite with cinnabar
            (17, 11, 54.5, 6.0e3, "chemical_reaction"), # Azurite with vermilion
            (17, 13, 58.9, 5.5e3, "chemical_reaction"), # Azurite with realgar
            (17, 23, 57.4, 5.6e3, "chemical_reaction"), # Azurite with orpiment
            (17, 24, 57.2, 5.5e3, "chemical_reaction"), # Azurite with arsenic sulfide
            (21, 10, 63.8, 4.2e3, "chemical_reaction"), # Atacamite with cinnabar
            (21, 11, 63.5, 4.1e3, "chemical_reaction"), # Atacamite with vermilion
            (22, 10, 56.3, 5.7e3, "chemical_reaction"), # Malachite with cinnabar
            (22, 11, 56.0, 5.6e3, "chemical_reaction"), # Malachite with vermilion
            
            # Arsenic-based pigments incompatibilities
            (13, 17, 59.4, 5.2e3, "chemical_reaction"), # Realgar with azurite
            (13, 21, 62.8, 4.7e3, "chemical_reaction"), # Realgar with atacamite
            (13, 22, 60.5, 5.0e3, "chemical_reaction"), # Realgar with malachite
            (23, 17, 58.7, 5.3e3, "chemical_reaction"), # Orpiment with azurite
            (23, 21, 62.3, 4.8e3, "chemical_reaction"), # Orpiment with atacamite
            (23, 22, 59.9, 5.1e3, "chemical_reaction"), # Orpiment with malachite
            
            # Organic and inorganic pigment incompatibilities (catalytic degradation)
            (30, 10, 68.5, 3.4e3, "catalytic_degradation"), # Indigo with cinnabar
            (32, 14, 71.2, 3.0e3, "catalytic_degradation"), # Lac with lead oxide
            (31, 23, 73.7, 2.8e3, "catalytic_degradation"), # Gamboge with orpiment
            
            # NEW: Additional metal-organic interactions
            (3, 30, 70.2, 3.1e3, "catalytic_degradation"),  # Lead white with indigo
            (3, 32, 69.8, 3.2e3, "catalytic_degradation"),  # Lead white with lac
            (17, 30, 67.4, 3.5e3, "catalytic_degradation"), # Azurite with indigo
            (17, 31, 68.1, 3.3e3, "catalytic_degradation"), # Azurite with gamboge
            
            # NEW: pH-related incompatibilities
            (12, 2, 64.5, 3.8e3, "acid_base_reaction"),     # Hematite with gypsum
            (0, 14, 65.8, 3.6e3, "acid_base_reaction"),     # Kaolin with lead oxide
            (17, 2, 63.2, 4.0e3, "acid_base_reaction"),     # Azurite with gypsum
            (22, 2, 62.9, 4.1e3, "acid_base_reaction"),     # Malachite with gypsum
        ]

    def _init_quantum_efficiency(self):
        """
        Initialize photodegradation quantum efficiency parameters.
        
        Format: pigment_id: {wavelength_range: quantum_efficiency}
        Quantum efficiency as photons that cause degradation / total absorbed photons
        """
        return {
            # Arsenic sulfides (highly light sensitive)
            13: {"uv_a": 0.0047, "uv_b": 0.0185, "visible": 0.0012},  # Realgar
            23: {"uv_a": 0.0043, "uv_b": 0.0178, "visible": 0.0010},  # Orpiment
            
            # Organic pigments (light sensitive)
            30: {"uv_a": 0.0038, "uv_b": 0.0172, "visible": 0.0008},  # Indigo
            31: {"uv_a": 0.0041, "uv_b": 0.0168, "visible": 0.0009},  # Gamboge
            32: {"uv_a": 0.0040, "uv_b": 0.0170, "visible": 0.0008},  # Lac
            33: {"uv_a": 0.0042, "uv_b": 0.0175, "visible": 0.0010},  # Phellodendron (NEW)
            
            # Moderately light sensitive
            10: {"uv_a": 0.0018, "uv_b": 0.0082, "visible": 0.0004},  # Cinnabar
            11: {"uv_a": 0.0017, "uv_b": 0.0080, "visible": 0.0004},  # Vermilion
            17: {"uv_a": 0.0019, "uv_b": 0.0085, "visible": 0.0005},  # Azurite
            22: {"uv_a": 0.0023, "uv_b": 0.0090, "visible": 0.0006},  # Malachite (NEW)
            
            # Lower sensitivity pigments
            3: {"uv_a": 0.0010, "uv_b": 0.0045, "visible": 0.0002},   # Lead white
            0: {"uv_a": 0.0005, "uv_b": 0.0020, "visible": 0.0001},   # Kaolin
            25: {"uv_a": 0.0002, "uv_b": 0.0010, "visible": 0.0001},  # Carbon black (NEW)
            28: {"uv_a": 0.0001, "uv_b": 0.0005, "visible": 0.0001},  # Gold leaf (NEW)
        }

    def _init_nanoscale_properties(self):
        """
        Initialize nanoscale physical properties.
        
        Format: pigment_id: {particle_size, surface_energy, zeta_potential, aggregation_tendency}
        """
        properties = {}

        for i in range(min(len(self.pigment_db.pigments), 35)):
            try:
                particle_data = self.pigment_db.get_particle_data(i)
                size = particle_data["size"]["mean"]
                properties[i] = {
                    "particle_size": size,
                    "surface_energy": 40 + 30 * (1/size), 
                    "zeta_potential": -20 - 10 * (1/size), 
                    "aggregation_tendency": 0.3 + 0.3 * (1/size)
                }
                
                if "microstructure" in particle_data and "porosity" in particle_data["microstructure"]:
                    porosity = particle_data["microstructure"]["porosity"]
                    properties[i]["surface_energy"] *= (1 + 0.5 * porosity)
                    properties[i]["aggregation_tendency"] *= (1 - 0.3 * porosity)
                    
            except (KeyError, IndexError):
                pass

        default_properties = {
            # White pigments
            0: {"particle_size": 2.8, "surface_energy": 38, "zeta_potential": -25, "aggregation_tendency": 0.3},   # Kaolin
            1: {"particle_size": 5.2, "surface_energy": 42, "zeta_potential": -18, "aggregation_tendency": 0.4},   # Calcium carbonate
            2: {"particle_size": 8.5, "surface_energy": 36, "zeta_potential": -22, "aggregation_tendency": 0.35},  # Gypsum
            3: {"particle_size": 1.2, "surface_energy": 58, "zeta_potential": -12, "aggregation_tendency": 0.6},   # Lead white
            
            # Red pigments
            10: {"particle_size": 0.8, "surface_energy": 68, "zeta_potential": -28, "aggregation_tendency": 0.45}, # Cinnabar
            11: {"particle_size": 0.9, "surface_energy": 67, "zeta_potential": -27, "aggregation_tendency": 0.46}, # Vermilion
            12: {"particle_size": 0.3, "surface_energy": 72, "zeta_potential": -32, "aggregation_tendency": 0.25}, # Hematite
            
            # Blue pigments
            17: {"particle_size": 3.8, "surface_energy": 52, "zeta_potential": -24, "aggregation_tendency": 0.5},  # Azurite
            18: {"particle_size": 12.5, "surface_energy": 45, "zeta_potential": -30, "aggregation_tendency": 0.3}, # Lapis lazuli
            
            # Copper greens
            21: {"particle_size": 2.9, "surface_energy": 55, "zeta_potential": -22, "aggregation_tendency": 0.55}, # Atacamite
            22: {"particle_size": 3.6, "surface_energy": 53, "zeta_potential": -23, "aggregation_tendency": 0.52}, # Malachite
            
            # Arsenic pigments
            13: {"particle_size": 1.8, "surface_energy": 62, "zeta_potential": -15, "aggregation_tendency": 0.65}, # Realgar
            23: {"particle_size": 1.6, "surface_energy": 60, "zeta_potential": -16, "aggregation_tendency": 0.62}, # Orpiment
        }

        for pid, props in default_properties.items():
            if pid not in properties:
                properties[pid] = props
        
        return properties

    def _init_microbial_susceptibility(self):
        """
        Initialize microbiological susceptibility.
        
        Format: pigment_id: {organic_content, nutrient_value, biocide_properties}
        Scale 0-1 where 1 is high susceptibility/organic content
        """
        return {
            # Organic pigments (high susceptibility)
            30: {"organic_content": 0.95, "nutrient_value": 0.85, "biocide_properties": 0.0},  # Indigo
            31: {"organic_content": 0.90, "nutrient_value": 0.80, "biocide_properties": 0.1},  # Gamboge
            32: {"organic_content": 0.92, "nutrient_value": 0.82, "biocide_properties": 0.0},  # Lac
            33: {"organic_content": 0.88, "nutrient_value": 0.78, "biocide_properties": 0.08}, # Phellodendron (NEW)
            
            # Inorganic with organic binders
            0: {"organic_content": 0.05, "nutrient_value": 0.10, "biocide_properties": 0.0},   # Kaolin
            1: {"organic_content": 0.02, "nutrient_value": 0.08, "biocide_properties": 0.0},   # Calcium carbonate
            2: {"organic_content": 0.03, "nutrient_value": 0.09, "biocide_properties": 0.0},   # Gypsum
            
            # Toxic pigments with biocidal properties
            3: {"organic_content": 0.01, "nutrient_value": 0.05, "biocide_properties": 0.7},   # Lead white
            10: {"organic_content": 0.0, "nutrient_value": 0.03, "biocide_properties": 0.85},  # Cinnabar (mercury)
            13: {"organic_content": 0.0, "nutrient_value": 0.02, "biocide_properties": 0.8},   # Realgar (arsenic)
            23: {"organic_content": 0.0, "nutrient_value": 0.02, "biocide_properties": 0.82},  # Orpiment (arsenic)
            14: {"organic_content": 0.0, "nutrient_value": 0.04, "biocide_properties": 0.75},  # Lead oxide (NEW)
        }

    def _init_ph_sensitivity(self):
        """
        Initialize pH sensitivity parameters.
        
        Format: pigment_id: {acid: sensitivity, alkaline: sensitivity, mechanism: description}
        """
        return {
            0: {"acid": 0.3, "alkaline": 0.2, "mechanism": "slow dissolution"},
            1: {"acid": 0.5, "alkaline": 0.2, "mechanism": "slow dissolution"},
            2: {"acid": 0.6, "alkaline": 0.3, "mechanism": "dissolution and recrystallization"},
            3: {"acid": 0.8, "alkaline": 0.1, "mechanism": "carbonate decomposition"},
            5: {"acid": 0.7, "alkaline": 0.2, "mechanism": "conversion to oxychloride"},
            9: {"acid": 0.6, "alkaline": 0.3, "mechanism": "conversion to oxide"},
            10: {"acid": 0.1, "alkaline": 0.1, "mechanism": "stable structure"},
            12: {"acid": 0.2, "alkaline": 0.1, "mechanism": "stable oxide structure"},
            13: {"acid": 0.3, "alkaline": 0.5, "mechanism": "sulfide conversion"},
            17: {"acid": 0.9, "alkaline": 0.1, "mechanism": "carbonate decomposition"},
            21: {"acid": 0.6, "alkaline": 0.3, "mechanism": "chloride displacement"},
            22: {"acid": 0.7, "alkaline": 0.2, "mechanism": "carbonate decomposition"},
            23: {"acid": 0.2, "alkaline": 0.6, "mechanism": "sulfide conversion"},
            24: {"acid": 0.3, "alkaline": 0.5, "mechanism": "sulfide conversion"},
            25: {"acid": 0.1, "alkaline": 0.1, "mechanism": "carbon stability"},
            30: {"acid": 0.4, "alkaline": 0.5, "mechanism": "chromophore degradation"},
            31: {"acid": 0.5, "alkaline": 0.6, "mechanism": "resin degradation"},
            32: {"acid": 0.6, "alkaline": 0.4, "mechanism": "dye structure breakdown"},
            33: {"acid": 0.5, "alkaline": 0.5, "mechanism": "alkaloid degradation"}
        }

    def _init_historical_data(self):
        """Initialize historical data with improved organization"""
        return {
            "early_tang": {
                "compatible_groups": [
                    [0, 1, 2],      # White pigment group
                    [10, 12, 14],   # Red pigment group
                    [17, 18],       # Blue pigment group
                    [25]            # Black
                ],
                "avoided_combinations": [
                    [3, 10], [3, 11], [17, 10], [17, 11]
                ]
            },
            "middle_tang": {
                "compatible_groups": [
                    [0, 1, 2, 3],    # White pigment group
                    [10, 12, 14],    # Red pigment group
                    [17, 18, 19],    # Blue pigment group
                    [21, 22],        # Green pigment group
                    [23, 24, 26],    # Yellow pigment group
                    [25, 27]         # Black and brown pigments
                ],
                "avoided_combinations": [
                    [3, 10], [3, 11], [17, 10], [17, 11], [22, 23]
                ]
            },
            "late_tang": {
                "compatible_groups": [
                    [0, 1, 2, 3, 4, 5],    # White pigment group
                    [10, 11, 12, 14, 32],  # Red pigment group with organics
                    [17, 18, 19, 30],      # Blue pigment group with indigo
                    [21, 22],              # Green pigment group
                    [23, 24, 26, 31, 33],  # Yellow pigment group with organics
                    [25, 27],              # Black and brown pigments
                    [28]                   # Gold
                ],
                "avoided_combinations": [
                    [3, 10], [3, 11], [17, 10], [17, 11], [22, 23], [30, 14]
                ]
            }
        }

    def _init_binder_compatibility(self):
        """Initialize binder compatibility data."""
        return {
            "animal_glue": {
                0: 0.9, 1: 0.9, 2: 0.9,     # White pigments highly compatible
                10: 0.7, 11: 0.7,           # Moderate with mercury pigments
                17: 0.8, 18: 0.8,           # Good with copper blues
                21: 0.7, 22: 0.7,           # Decent with copper greens
                25: 0.9,                    # Excellent with carbon black
                23: 0.6, 24: 0.6,           # Fair with arsenic pigments (NEW)
                30: 0.8, 31: 0.7, 32: 0.8   # Good with most organics (NEW)
            },
            "gum_arabic": {
                0: 0.8, 1: 0.8, 2: 0.8,     # Good with white pigments
                10: 0.7, 11: 0.7,           # Moderate with mercury pigments
                17: 0.9, 18: 0.9,           # Excellent with copper blues
                23: 0.9, 24: 0.9,           # Excellent with arsenic yellows
                30: 0.9, 31: 0.9, 32: 0.9,  # Excellent with organics
                21: 0.8, 22: 0.8            # Good with copper greens (NEW)
            },
            "drying_oil": {
                0: 0.6, 1: 0.6, 2: 0.5,     # Moderate with whites
                3: 0.9, 5: 0.8, 9: 0.8,     # Excellent with lead pigments
                10: 0.8, 11: 0.8,           # Good with mercury pigments
                12: 0.9, 14: 0.9,           # Excellent with iron oxide and lead oxide
                17: 0.6, 18: 0.5,           # Moderate with copper blues
                21: 0.6, 22: 0.5,           # Moderate with copper greens
                30: 0.5, 31: 0.5, 32: 0.5   # Fair with organics (NEW)
            },
            "egg_tempera": {
                0: 0.8, 1: 0.8, 2: 0.7,     # Good with whites
                3: 0.9, 5: 0.9, 9: 0.9,     # Excellent with lead pigments
                10: 0.8, 11: 0.8,           # Good with mercury pigments
                17: 0.8, 18: 0.7,           # Good with copper blues
                21: 0.7, 22: 0.7,           # Good with copper greens
                30: 0.8, 31: 0.8, 32: 0.8   # Good with organics
            }
        }

    def _init_spectral_data(self):
        """
        Initialize spectral reflectance data for pigments.
        
        This uses the data from DunhuangPigmentDB for consistency across modules
        """
        spectral_data = {}
        for i, pigment in enumerate(self.pigment_db.pigments):
            try:
                wavelengths, reflectance = self.pigment_db.get_spectral_reflectance(i)
                
                spectral_data[i] = {
                    'wavelengths': wavelengths,
                    'reflectance': reflectance
                }
            except Exception as e:
                default_reflectance = np.ones(31) * 0.5
                spectral_data[i] = {
                    'wavelengths': np.linspace(400, 700, 31),
                    'reflectance': default_reflectance.tolist()
                }
                    
        return spectral_data
    
    def _calculate_ciede2000(self, lab1, lab2):
        """
        Calculate CIEDE2000 color difference.
        
        Args:
            lab1, lab2: Lab color coordinates [L, a, b]
            
        Returns:
            CIEDE2000 color difference
        """
        L1, a1, b1 = lab1
        L2, a2, b2 = lab2
        
        # Step 1: Calculate C1, C2, C_mean
        C1 = np.sqrt(a1**2 + b1**2)
        C2 = np.sqrt(a2**2 + b2**2)
        C_mean = (C1 + C2) / 2
        
        # Step 2: Calculate G to adjust a* values
        G = 0.5 * (1 - np.sqrt(C_mean**7 / (C_mean**7 + 25**7)))
        
        # Step 3: Calculate a', C', h' values
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        C1_prime = np.sqrt(a1_prime**2 + b1**2)
        C2_prime = np.sqrt(a2_prime**2 + b2**2)
        C_mean_prime = (C1_prime + C2_prime) / 2
        
        # Calculate h' values (hue angles) with proper handling of edge cases
        h1_prime = np.arctan2(b1, a1_prime) 
        h2_prime = np.arctan2(b2, a2_prime)
        
        # Convert to degrees
        h1_prime = np.degrees(h1_prime)
        h2_prime = np.degrees(h2_prime)
        
        # Ensure positive angles
        if h1_prime < 0:
            h1_prime += 360
        if h2_prime < 0:
            h2_prime += 360
        
        # Step 4: Calculate ΔL', ΔC', ΔH'
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        
        # Calculate changes in h'
        if C1_prime * C2_prime == 0:
            delta_h_prime = 0
        else:
            delta_h_prime = h2_prime - h1_prime
            # Adjust for angles > 180 degrees
            if delta_h_prime > 180:
                delta_h_prime -= 360
            elif delta_h_prime < -180:
                delta_h_prime += 360
        
        # Calculate changes in H'
        delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime/2))
        
        # Step 5: Calculate CIEDE2000 weighting functions
        # Lightness
        SL = 1 + (0.015 * (L_mean := (L1 + L2) / 2 - 50)**2) / np.sqrt(20 + L_mean**2)
        
        # Chroma
        SC = 1 + 0.045 * C_mean_prime
        
        # Hue
        T = 1 - 0.17 * np.cos(np.radians(h_mean_prime := (
                h1_prime + h2_prime + (360 if abs(h1_prime - h2_prime) > 180 else 0)
            ) / 2 - 30)) \
            + 0.24 * np.cos(np.radians(2 * h_mean_prime)) \
            + 0.32 * np.cos(np.radians(3 * h_mean_prime + 6)) \
            - 0.20 * np.cos(np.radians(4 * h_mean_prime - 63))
        SH = 1 + 0.015 * C_mean_prime * T
        
        # Rotation term
        RC = 2 * np.sqrt(C_mean_prime**7 / (C_mean_prime**7 + 25**7))
        delta_theta = 30 * np.exp(-((h_mean_prime - 275) / 25)**2)
        RT = -RC * np.sin(np.radians(2 * delta_theta))
        
        # Step 6: Calculate the final CIEDE2000 color difference
        delta_E = np.sqrt(
            (delta_L_prime / SL)**2 +
            (delta_C_prime / SC)**2 +
            (delta_H_prime / SH)**2 +
            RT * (delta_C_prime / SC) * (delta_H_prime / SH)
        )
        
        return delta_E
    
    def _rgb_to_lab(self, rgb):
        """
        Convert RGB to Lab color space.
        
        Args:
            rgb: RGB color values [r, g, b] in range [0,1]
            
        Returns:
            Lab color values [L, a, b]
        """
        # Step 1: Convert RGB [0,1] to sRGB
        r, g, b = rgb
        
        # Apply gamma correction
        def gamma_correct(c):
            return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1/2.4)) - 0.055
        
        r = gamma_correct(r)
        g = gamma_correct(g)
        b = gamma_correct(b)
        
        # Step 2: sRGB to XYZ conversion (using D65 white point matrix)
        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
        
        # Step 3: XYZ to Lab
        xn, yn, zn = 0.95047, 1.0, 1.08883
        
        def f(t):
            return t**(1/3) if t > 0.008856 else 7.787*t + 16/116
        
        fx = f(x / xn)
        fy = f(y / yn)
        fz = f(z / zn)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return [L, a, b]

    def _spectrum_to_rgb(self, spectrum):
        """
        Convert spectral data to RGB.
        
        Args:
            spectrum: Spectral reflectance array
            
        Returns:
            RGB color values [r, g, b]
        """
        if isinstance(spectrum, list):
            spectrum = np.array(spectrum)
        
        # Use CIE standard observer functions
        wavelengths = np.linspace(400, 700, len(spectrum))
        
        # CIE 1931 standard observer color matching functions (x, y, z)
        x_bar = 1.056 * np.exp(-0.5 * ((wavelengths - 599) / 37.9)**2) + \
                0.362 * np.exp(-0.5 * ((wavelengths - 442) / 16.0)**2) + \
                0.065 * np.exp(-0.5 * ((wavelengths - 513) / 31.1)**2)
        
        y_bar = 1.062 * np.exp(-0.5 * ((wavelengths - 558) / 46.9)**2) + \
                0.040 * np.exp(-0.5 * ((wavelengths - 609) / 31.0)**2)
        
        z_bar = 0.980 * np.exp(-0.5 * ((wavelengths - 446) / 19.5)**2) + \
                0.062 * np.exp(-0.5 * ((wavelengths - 513) / 26.6)**2)
        
        # Calculate XYZ tristimulus values
        X = np.sum(spectrum * x_bar) / np.sum(y_bar)
        Y = np.sum(spectrum * y_bar) / np.sum(y_bar)
        Z = np.sum(spectrum * z_bar) / np.sum(y_bar)
        
        # Convert XYZ to RGB using sRGB transformation matrix
        r = max(0, min(1,  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z))
        g = max(0, min(1, -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z))
        b = max(0, min(1,  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z))
        
        # Convert linear RGB to gamma-corrected sRGB
        def inverse_gamma(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        
        r = inverse_gamma(r)
        g = inverse_gamma(g)
        b = inverse_gamma(b)
        
        return [r, g, b]
    
    def _calculate_reaction_rate(self, pigment1, pigment2, temperature=298):
        """
        Calculate reaction rate using Arrhenius equation.
        k = A * exp(-Ea / (R * T))
        
        Args:
            pigment1, pigment2: Pigment IDs
            temperature: Temperature in K (default 298K = 25°C)
            
        Returns:
            Reaction rate and reaction type if reaction exists, (0, None) otherwise
        """
        # Gas constant (R) in J/mol.K
        R = 8.314

        # Convert IDs to scalar integers
        pigment1 = self._extract_scalar(pigment1)
        pigment2 = self._extract_scalar(pigment2)
        
        # Check for reaction in both directions
        for p1, p2, Ea, A, reaction_type in self.reaction_parameters:
            if (pigment1 == p1 and pigment2 == p2) or (pigment1 == p2 and pigment2 == p1):
                # Ea is in kJ/mol, convert to J/mol
                Ea_J = abs(Ea * 1000)
                # Calculate rate constant with proper units
                k = A * np.exp(-Ea_J / (R * temperature))
                return float(k), reaction_type

        return 0, None
    
    def _calculate_nanoscale_interaction(self, pigment1, pigment2):
        """
        Calculate nanoscale interaction energy between pigment particles.
        
        Args:
            pigment1, pigment2: Pigment IDs
            
        Returns:
            Interaction energy and stability impact
        """
        if hasattr(pigment1, 'item'):
            pigment1 = pigment1.item()
        if hasattr(pigment2, 'item'):
            pigment2 = pigment2.item()
        if pigment1 not in self.nanoscale_properties or pigment2 not in self.nanoscale_properties:
            return 0, 0 
            
        p1 = self.nanoscale_properties[pigment1]
        p2 = self.nanoscale_properties[pigment2]
        size_ratio = max(p1["particle_size"], p2["particle_size"]) / min(p1["particle_size"], p2["particle_size"])
        surface_energy_diff = abs(p1["surface_energy"] - p2["surface_energy"])
        zeta_diff = abs(p1["zeta_potential"] - p2["zeta_potential"])
        aggregation_potential = (p1["aggregation_tendency"] + p2["aggregation_tendency"]) / 2
        interaction_energy = (
            (0.4 * size_ratio) +                
            (0.3 * surface_energy_diff / 50) + 
            (0.1 * (1 - zeta_diff / 40)) + 
            (0.2 * aggregation_potential) 
        )
        stability_impact = 1.0 - np.exp(-interaction_energy / 1.5)
        stability_impact = min(1.0, max(0.0, stability_impact))
        
        return interaction_energy, stability_impact
    
    def _calculate_microbial_risk(self, pigment_indices, environmental_conditions=None):
        """
        Calculate risk of microbiological degradation.
        
        Args:
            pigment_indices: List of pigment IDs
            environmental_conditions: Environmental factors
            
        Returns:
            Microbial risk score (0-1) and detailed assessment
        """
        safe_indices = []
        for p in pigment_indices:
            if hasattr(p, 'item'):
                if hasattr(p, 'numel') and p.numel() > 1:
                    safe_indices.append(p[0].item())
                elif hasattr(p, 'size') and p.size > 1:
                    safe_indices.append(p[0].item())
                else:
                    safe_indices.append(p.item())
            else:
                safe_indices.append(p)

        pigment_indices = safe_indices

        if not environmental_conditions or "humidity" not in environmental_conditions:
            return 0.1, {"risk_level": "Low", "reason": "Insufficient environmental data"}
        humidity = environmental_conditions.get("humidity", 0)
        if humidity < 0.6:
            return 0.2, {"risk_level": "Low", "reason": "Humidity too low for significant microbial activity"}
        
        copper_based = any(p in [17, 21, 22] for p in pigment_indices) 
        if copper_based and humidity > 0.6:
            base_risk = 0.3 + humidity * 0.4
            return base_risk, {"risk_level": "Moderate to High", 
                            "reason": "Copper compounds sensitive to humidity"}
        
        total_organic = 0
        total_nutrients = 0
        total_biocide = 0
        count = 0
        
        for pigment_id in pigment_indices:
            if pigment_id in self.microbial_susceptibility:
                data = self.microbial_susceptibility[pigment_id]
                total_organic += data["organic_content"]
                total_nutrients += data["nutrient_value"]
                total_biocide += data["biocide_properties"]
                count += 1
                
        if count == 0:
            return 0.3, {"risk_level": "Low-Moderate", "reason": "Unknown pigment susceptibility"}
            
        avg_organic = total_organic / count
        avg_nutrients = total_nutrients / count
        avg_biocide = total_biocide / count

        humidity_factor = 0.5 * np.exp(humidity - 0.6)
        base_risk = ((0.5 * avg_organic) + (0.5 * avg_nutrients)) * humidity_factor
        biocide_protection = 1.0 - np.sqrt(avg_biocide)
        mitigated_risk = base_risk * biocide_protection
        risk_score = min(1.0, mitigated_risk)
        if risk_score < 0.3:
            risk_level = "Low"
            reason = "Minimal organic content and/or strong biocidal properties"
        elif risk_score < 0.6:
            risk_level = "Moderate"
            reason = "Some organic components with moderate humidity levels"
        else:
            risk_level = "High"
            reason = "High organic content and humidity, limited biocidal properties"
            
        return risk_score, {"risk_level": risk_level, "reason": reason}
    
    def _calculate_quantum_efficiency_factor(self, pigment_id, light_exposure):
        """
        Calculate light degradation factor based on quantum efficiency.
        
        Args:
            pigment_id: Pigment ID
            light_exposure: Light exposure level (0-1)
            
        Returns:
            Degradation factor and detailed information
        """
        pigment_id = self._extract_scalar(pigment_id)
        aging_factors = self.pigment_db.get_aging_factors(pigment_id)

        if isinstance(aging_factors, dict):
            if isinstance(aging_factors.get("fading", 0), dict):
                base_fading = aging_factors["fading"].get("rate", 0.01)
                max_fading = aging_factors["fading"].get("max", 0.2)
            else:
                base_fading = aging_factors.get("fading", 0.01)
                max_fading = base_fading * 20 
        else:
            base_fading = 0.01
            max_fading = 0.2
        
        qe_values = {
            "uv_a": base_fading * 3.8,     
            "uv_b": base_fading * 18.5,   
            "visible": base_fading * 0.8 
        }
        
        mechanisms = {
            "uv_b": "UV-B photolysis",
            "uv_a": "UV-A induced oxidation",
            "visible": "visible light photochemical degradation"
        }
        primary = max(qe_values, key=qe_values.get)
        mechanism = mechanisms[primary]

        if light_exposure < 0.4:  
            uv_a_weight, uv_b_weight, visible_weight = 0.2, 0.1, 0.7
        elif light_exposure < 0.7:  
            uv_a_weight, uv_b_weight, visible_weight = 0.3, 0.2, 0.5
        else:  
            uv_a_weight, uv_b_weight, visible_weight = 0.4, 0.3, 0.3
            
        degradation_factor = (
            uv_a_weight * qe_values["uv_a"] +
            uv_b_weight * qe_values["uv_b"] +
            visible_weight * qe_values["visible"]
        ) * light_exposure * 50
        
        if light_exposure > 0.7:
            degradation_factor *= (1 + (light_exposure - 0.7) * 0.5)

        degradation_factor = min(max_fading, 1.0 - np.exp(-degradation_factor))
        
        return min(1.0, degradation_factor), {"mechanism": mechanism}
    
    def check_mixture_stability(self, pigment_indices, mixing_ratios, 
                           binder_type=None, environmental_conditions=None,
                           temperature=298):
        """
        Perform comprehensive stability analysis of pigment mixture.
        
        Args:
            pigment_indices: Array of pigment indices
            mixing_ratios: Array of mixing ratios
            binder_type: Optional binder material
            environmental_conditions: Optional dict with environmental factors
            temperature: Temperature in K (default 298K = 25°C)
            
        Returns:
            Dictionary with detailed stability information
        """
        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().numpy()
        else:
            indices = np.array(pigment_indices)
        if isinstance(mixing_ratios, torch.Tensor):
            ratios = mixing_ratios.detach().cpu().numpy()
        else:
            ratios = np.array(mixing_ratios)
        ratios = ratios / np.sum(ratios)
        
        base_stability_score = 1.0
        warnings = []
        unstable_pairs = []
        detailed_issues = []
    
        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3
            }
        
        # 1. Check for chemical reactions between pigments using Arrhenius kinetics
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                pigment1 = indices[i]
                pigment2 = indices[j]

                current_ratio_i, current_ratio_j = ratios[i], ratios[j]
                if isinstance(current_ratio_i, np.ndarray):
                    current_ratio_i = np.mean(current_ratio_i)
                if isinstance(current_ratio_j, np.ndarray):
                    current_ratio_j = np.mean(current_ratio_j)
                    
                if current_ratio_i < 0.03 or current_ratio_j < 0.03:
                    continue


                rate, reaction_type = self._calculate_reaction_rate(pigment1, pigment2, temperature)
                
                if rate > 0:
                    concentration_factor = min(ratios[i], ratios[j])
                    if reaction_type == "chemical_reaction":
                        severity = min(1.0, rate / 1e3) 
                    elif reaction_type == "catalytic_degradation":
                        severity = min(1.0, rate / 5e2)  
                    else:
                        severity = min(1.0, rate / 2e3) 

                    instability = concentration_factor * severity * 0.5
                    base_stability_score -= instability

                    p1_name = self.pigment_db.id_to_name.get(pigment1, f"Pigment {pigment1}")
                    p2_name = self.pigment_db.id_to_name.get(pigment2, f"Pigment {pigment2}")

                    reaction_desc = "Unknown reaction"
                    for p1_idx, p2_idx, _, _, rxn_type in self.reaction_parameters:
                        if (pigment1 == p1_idx and pigment2 == p2_idx) or (pigment1 == p2_idx and pigment2 == p1_idx):
                            if rxn_type == "chemical_reaction":
                                reaction_desc = f"Forms {self._get_reaction_product(pigment1, pigment2)}"
                            elif rxn_type == "catalytic_degradation":
                                reaction_desc = "Causes catalytic degradation"
                            break
                    
                    warning = f"Chemical incompatibility: {p1_name} and {p2_name}. {reaction_desc}"
                    warnings.append(warning)
                    
                    unstable_pairs.append((i, j))
                    detailed_issues.append({
                        "type": "chemical_reaction",
                        "pigments": [p1_name, p2_name],
                        "mechanism": reaction_type,
                        "rate_constant": rate,
                        "severity": severity,
                        "description": reaction_desc,
                        "impact": instability
                    })
                    
                # 2. Check for nanoscale physical interactions
                energy, impact = self._calculate_nanoscale_interaction(pigment1, pigment2)
                
                if impact > 0.3: 
                    p1_name = self.pigment_db.id_to_name.get(pigment1, f"Pigment {pigment1}")
                    p2_name = self.pigment_db.id_to_name.get(pigment2, f"Pigment {pigment2}")

                    nano_instability = impact * 0.3 * min(ratios[i], ratios[j])
                    base_stability_score -= nano_instability

                    if abs(self.nanoscale_properties[pigment1]["particle_size"] - 
                         self.nanoscale_properties[pigment2]["particle_size"]) > 5:
                        issue_type = "particle size mismatch"
                    elif abs(self.nanoscale_properties[pigment1]["surface_energy"] - 
                          self.nanoscale_properties[pigment2]["surface_energy"]) > 20:
                        issue_type = "surface energy incompatibility"
                    else:
                        issue_type = "particle aggregation tendency"
                    
                    warning = f"Nanoscale interaction issue: {p1_name} and {p2_name} - {issue_type}"
                    warnings.append(warning)
                    
                    detailed_issues.append({
                        "type": "nanoscale_interaction",
                        "pigments": [p1_name, p2_name],
                        "interaction_energy": energy,
                        "issue": issue_type,
                        "impact": nano_instability
                    })
        
        # 3. Check photodegradation based on quantum efficiency
        if "light_exposure" in environmental_conditions and environmental_conditions["light_exposure"] > 0.2:
            light_exposure = environmental_conditions["light_exposure"]
            
            for i, pigment_id in enumerate(indices):
                current_ratio = ratios[i]
                if isinstance(current_ratio, np.ndarray):
                    current_ratio = np.mean(current_ratio)
                
                if current_ratio < 0.05:
                    continue
                    
                degradation_result = self._calculate_quantum_efficiency_factor(
                    pigment_id, light_exposure
                )

                if isinstance(degradation_result, tuple):
                    degradation_factor, degradation_info = degradation_result
                else:
                    degradation_factor = degradation_result
                    degradation_info = {"mechanism": "photodegradation"}
                
                if degradation_factor > 0.2:  
                    current_ratio = ratios[i]
                    if isinstance(current_ratio, np.ndarray):
                        current_ratio = np.mean(current_ratio)
                    photo_impact = degradation_factor * current_ratio * 0.4
                    base_stability_score -= photo_impact
                    
                    pid = self._extract_scalar(pigment_id)
                    p_name = self.pigment_db.id_to_name.get(pid, f"Pigment {pid}")
                    warning = f"Photodegradation risk: {p_name}"
                    warnings.append(warning)
                    
                    detailed_issues.append({
                        "type": "photodegradation",
                        "pigment": p_name,
                        "degradation_factor": degradation_factor,
                        "impact": photo_impact,
                        "mechanism": degradation_info.get("mechanism", "photodegradation")
                    })
        
        # 4. Check microbiological risks
        microbial_risk, risk_info = self._calculate_microbial_risk(indices, environmental_conditions)
        
        if microbial_risk > 0.4:  
            microbial_impact = microbial_risk * 0.3
            base_stability_score -= microbial_impact
            
            warning = f"Microbiological degradation risk: {risk_info['risk_level']} - {risk_info['reason']}"
            warnings.append(warning)
            
            detailed_issues.append({
                "type": "microbial_risk",
                "risk_level": risk_info["risk_level"],
                "reason": risk_info["reason"],
                "impact": microbial_impact
            })
        
        # 5. Check binder compatibility if provided
        binder_penalty = 0
        if binder_type and binder_type in self.binder_compatibility:
            binder_data = self.binder_compatibility[binder_type]
            
            for i, pigment_id in enumerate(indices):
                current_ratio = ratios[i]
                if isinstance(current_ratio, np.ndarray):
                    current_ratio = np.mean(current_ratio)
                    
                if current_ratio < 0.05:
                    continue

                if hasattr(pigment_id, 'item'):
                    pigment_scalar = pigment_id.item() if pigment_id.size == 1 else pigment_id[0].item()
                elif isinstance(pigment_id, np.ndarray):
                    pigment_scalar = pigment_id.item() if pigment_id.size == 1 else pigment_id[0]
                else:
                    pigment_scalar = pigment_id
                
                if pigment_scalar in binder_data:
                    compatibility = binder_data[pigment_scalar]
                    if compatibility < 0.7:
                        current_ratio = ratios[i]
                        if isinstance(current_ratio, np.ndarray):
                            current_ratio = np.mean(current_ratio)
                        penalty = (0.7 - compatibility) * current_ratio * 0.5
                        binder_penalty += penalty
                        
                        p_name = self.pigment_db.id_to_name.get(pigment_scalar, f"Pigment {pigment_scalar}")
                        warnings.append(f"Poor compatibility between {p_name} and {binder_type} binder")
                        detailed_issues.append({
                            "type": "binder_incompatibility",
                            "pigment": p_name,
                            "binder": binder_type,
                            "compatibility": compatibility,
                            "impact": penalty
                        })
            
            base_stability_score -= binder_penalty
            
        # 6. Check for pH risks
        total_acidity = 0
        total_alkalinity = 0
        for i, pigment_id in enumerate(indices):
            pid = self._extract_scalar(pigment_id)
            
            if pid in self.ph_sensitivity:
                sens_data = self.ph_sensitivity[pid]
                if sens_data["acid"] > 0.5:
                    total_acidity += sens_data["acid"] * ratios[i]
                if sens_data["alkaline"] > 0.5:
                    total_alkalinity += sens_data["alkaline"] * ratios[i]

        if safe_compare(total_acidity, 0.2, 'gt') and safe_compare(total_alkalinity, 0.2, 'gt'):
            ph_penalty = min(total_acidity, total_alkalinity) * 0.5
            base_stability_score -= ph_penalty
            warnings.append("Mixture contains both acidic and alkaline components")
            detailed_issues.append({
                "type": "ph_incompatibility",
                "acid_level": total_acidity,
                "alkaline_level": total_alkalinity,
                "impact": ph_penalty
            })
            
        # 7. Check compatibility with historical practices
        historical_bonus = 0
        for period, data in self.historical_data.items():
            for group in data["compatible_groups"]:
                match_count = 0
                for p in indices:
                    if hasattr(p, 'item'):
                        p_scalar = p.item() if p.size == 1 else p[0].item()
                    elif isinstance(p, np.ndarray):
                        p_scalar = p.item() if p.size == 1 else p[0]
                    else:
                        p_scalar = p
                        
                    if p_scalar in group:
                        match_count += 1
                        
                if match_count > 1 and match_count == len(indices):
                    historical_bonus = 0.1 
                    detailed_issues.append({
                        "type": "historical_match",
                        "period": period,
                        "impact": historical_bonus
                    })

            for avoided in data["avoided_combinations"]:
                if avoided[0] in indices and avoided[1] in indices:
                    warnings.append(f"This combination was historically avoided in {period} period")
                    detailed_issues.append({
                        "type": "historical_warning",
                        "period": period,
                        "avoided_pair": [
                            self.pigment_db.id_to_name.get(avoided[0], f"Pigment {avoided[0]}"),
                            self.pigment_db.id_to_name.get(avoided[1], f"Pigment {avoided[1]}")
                        ],
                        "impact": -0.1
                    })
                    historical_bonus -= 0.1
                    
        base_stability_score += historical_bonus
        
        # 8. Perform spectral verification if this is a color mixture
        if len(indices) > 1:
            spectral_stability = self._validate_spectral_stability(indices, ratios)
            
            if spectral_stability["delta_e"] > 5.0: 
                spectral_penalty = min(0.3, spectral_stability["delta_e"] / 30.0)
                base_stability_score -= spectral_penalty
                
                warnings.append(f"Expected color instability: {spectral_stability['description']}")
                detailed_issues.append({
                    "type": "spectral_instability",
                    "delta_e": spectral_stability["delta_e"],
                    "description": spectral_stability["description"],
                    "impact": spectral_penalty
                })

        if len(detailed_issues) > 1:
            impacts = [issue.get('impact', 0) for issue in detailed_issues]
            impacts.sort(reverse=True)

            synergy_penalty = sum(abs(impact)**1.2 for impact in impacts) - sum(impacts)
            issue_count_factor = 1.0 + (len(detailed_issues) - 1) * 0.1
            synergy_penalty *= issue_count_factor
            synergy_penalty = min(0.3, max(0, synergy_penalty)) 

            base_stability_score -= synergy_penalty

            if synergy_penalty > 0.05:
                warnings.append("Multiple issues have synergistic destabilizing effects")
                detailed_issues.append({
                    "type": "synergistic_effects",
                    "issue_count": len(detailed_issues),
                    "description": "Multiple incompatibilities amplify each other",
                    "impact": synergy_penalty
                })
        
        final_stability_score = max(0.0, min(1.0, float(abs(base_stability_score))))
        if final_stability_score >= 0.85:
            stability_rating = "Excellent"
        elif final_stability_score >= 0.7:
            stability_rating = "Good"
        elif final_stability_score >= 0.5:
            stability_rating = "Moderate"
        elif final_stability_score >= 0.3:
            stability_rating = "Poor"
        else:
            stability_rating = "Very Poor"
            
        return {
            "stability_score": final_stability_score,
            "stability_rating": stability_rating,
            "warnings": warnings,
            "unstable_pairs": unstable_pairs,
            "detailed_issues": detailed_issues,
            "is_stable": final_stability_score >= 0.7,
            "contribution_factors": {
                "base_chemical_compatibility": base_stability_score,
                "binder_penalty": binder_penalty if binder_type else 0,
                "historical_bonus": historical_bonus
            }
        }
    
    def _get_reaction_product(self, pigment1, pigment2):
        """Identify the reaction product between two pigments."""
        # This is a simplified lookup, I will add more in the future, if have time......
        if (pigment1 == 3 and pigment2 in [10, 11, 13]) or (pigment2 == 3 and pigment1 in [10, 11, 13]):
            return "black lead sulfide"
        elif (pigment1 in [17, 21, 22] and pigment2 in [10, 11, 13, 23, 24]) or \
             (pigment2 in [17, 21, 22] and pigment1 in [10, 11, 13, 23, 24]):
            return "copper sulfide (black)"
        else:
            return "degradation products"
    
    def _validate_spectral_stability(self, pigment_indices, mixing_ratios):
        """Validate spectral stability of mixture."""
        mixed_spectrum = np.zeros(31)

        if isinstance(pigment_indices, torch.Tensor):
            pigment_indices = pigment_indices.detach().cpu().numpy()
        if isinstance(mixing_ratios, torch.Tensor):
            mixing_ratios = mixing_ratios.detach().cpu().numpy()

        for i, pigment_id in enumerate(pigment_indices):
            if pigment_id in self.spectral_data:
                reflectance = np.array(self.spectral_data[pigment_id]['reflectance'])
                weighted_spectrum = reflectance * mixing_ratios[i]
                mixed_spectrum += weighted_spectrum

        if np.max(mixed_spectrum) > 1.0:
            mixed_spectrum = mixed_spectrum / np.max(mixed_spectrum)

        aged_spectrum = mixed_spectrum.copy()
        has_lead = any(p in [3, 5, 9, 14] for p in pigment_indices)
        has_sulfur = any(p in [10, 11, 13, 23, 24] for p in pigment_indices)
        has_copper = any(p in [17, 21, 22] for p in pigment_indices)
        
        if has_lead and has_sulfur:
            darkening_factor = 0.5 
            yellowing_factor = 0.2
        elif has_copper and has_sulfur:
            darkening_factor = 0.4
            yellowing_factor = 0.15
        else:
            darkening_factor = 0.2
            yellowing_factor = 0.15

        wavelengths = np.linspace(400, 700, 31)
        aged_spectrum *= (1.0 - darkening_factor)

        blue_region = wavelengths < 480
        yellow_region = (wavelengths >= 550) & (wavelengths < 600)
        
        if np.any(blue_region):
            aged_spectrum[blue_region] *= (1.0 - yellowing_factor * 1.5)
        if np.any(yellow_region):
            aged_spectrum[yellow_region] *= (1.0 + yellowing_factor)
        
        aged_spectrum = np.clip(aged_spectrum, 0.0, 1.0)
        original_rgb = self._spectrum_to_rgb(mixed_spectrum)
        aged_rgb = self._spectrum_to_rgb(aged_spectrum)
        original_lab = self._rgb_to_lab(original_rgb)
        aged_lab = self._rgb_to_lab(aged_rgb)
        delta_e = self._calculate_ciede2000(original_lab, aged_lab)

        if delta_e < 1.0:
            description = "Minimal color change expected"
        elif delta_e < 3.0:
            description = "Slight color shift over time"
        elif delta_e < 6.0:
            description = "Noticeable color change will develop"
        elif delta_e < 12.0:
            description = "Significant color transformation expected"
        else:
            description = "Dramatic color alteration will occur over time"
        
        if has_lead and has_sulfur:
            description += "; lead-sulfur reaction will cause blackening"
        elif has_copper and has_sulfur:
            description += "; copper-sulfur reaction will cause darkening"
        
        return {
            "original_spectrum": mixed_spectrum,
            "aged_spectrum": aged_spectrum,
            "original_rgb": original_rgb,
            "aged_rgb": aged_rgb,
            "delta_e": delta_e,
            "description": description
        }
    
    def analyze_painted_layers(self, layer_pigments, layer_binders=None, layer_order=None):
        """
        Analyze stability of multiple painted layers.
        
        Args:
            layer_pigments: List of lists of pigment indices for each layer
            layer_binders: Optional list of binder types for each layer
            layer_order: Optional specific ordering of layers (bottom to top)
            
        Returns:
            Dictionary with layer interaction analysis
        """
        if layer_order is None:
            layer_order = list(range(len(layer_pigments)))
        if layer_binders is None:
            layer_binders = ["animal_glue"] * len(layer_pigments)
        layer_stability = []
        for i, (pigments, binder) in enumerate(zip(layer_pigments, layer_binders)):
            ratios = np.ones(len(pigments)) / len(pigments)
            stability = self.check_mixture_stability(
                pigments, ratios, binder_type=binder
            )
            
            layer_stability.append({
                "layer_index": i,
                "layer_position": layer_order[i],
                "stability": stability
            })

        interlayer_issues = []
        for i in range(len(layer_pigments)):
            for j in range(i+1, len(layer_pigments)):
                bottom_idx = min(layer_order[i], layer_order[j])
                top_idx = max(layer_order[i], layer_order[j])
                bottom_layer = layer_pigments[layer_order.index(bottom_idx)]
                top_layer = layer_pigments[layer_order.index(top_idx)]
                issues = self._check_interlayer_interactions(
                    bottom_layer, top_layer, 
                    layer_binders[layer_order.index(bottom_idx)],
                    layer_binders[layer_order.index(top_idx)]
                )
                
                if issues:
                    interlayer_issues.append({
                        "bottom_layer": bottom_idx,
                        "top_layer": top_idx,
                        "issues": issues
                    })
        
        layer_scores = [layer["stability"]["stability_score"] for layer in layer_stability]

        weighted_scores = []
        for i, score in enumerate(layer_scores):
            layer_pos = layer_order[i]
            weight = 1.0 - (layer_pos / len(layer_scores)) * 0.3
            weighted_scores.append(score * weight)
            
        base_score = sum(weighted_scores) / sum(1.0 - (i / len(layer_scores)) * 0.3 
                                            for i in range(len(layer_scores)))

        interface_penalty = 0
        for issue in interlayer_issues:
            for problem in issue["issues"]:
                if problem["type"] == "layer_interaction":
                    interface_penalty += problem["severity"] * 0.15
                elif problem["type"] == "binder_interaction":
                    interface_penalty += problem["severity"] * 0.1

        if len(interlayer_issues) > 1:
            interface_penalty *= (1 + 0.2 * (len(interlayer_issues) - 1))
            
        interface_penalty = min(0.5, interface_penalty)

        diffusion_results = self._simulate_interlayer_diffusion(
            layer_pigments, 
            layer_binders, 
            years=50, 
            environmental_conditions={"humidity": 0.5, "temperature": 298, "temperature_fluctuation": 0.2}
        )

        diffusion_penalty = 0.0
        for result in diffusion_results:
            diffusion_penalty += result["stability_impact"]["impact_score"] * 0.2

        overall_score = base_score - interface_penalty - min(0.3, diffusion_penalty) 
        overall_score = max(0.0, min(1.0, overall_score))

        layer_analysis = {
            "layer_stability": layer_stability,
            "interlayer_issues": interlayer_issues,
            "overall_stability_score": overall_score,
            "is_stable": overall_score >= 0.65,
            "interface_penalty": interface_penalty,
            "interlayer_diffusion": diffusion_results,
            "diffusion_penalty": diffusion_penalty
        }
        
        return layer_analysis
    
    def monte_carlo_aging_simulation(self, pigment_indices, mixing_ratios=None, years=100, 
                               n_samples=100, parameter_bounds=None):
        """
        Run Monte Carlo simulation to quantify uncertainty in aging predictions.
        
        Args:
            pigment_indices: Pigment indices
            mixing_ratios: Optional mixing ratios
            years: Simulation time in years
            n_samples: Number of Monte Carlo samples
            parameter_bounds: Optional dictionary defining parameter ranges
            
        Returns:
            Dictionary with statistical results of multiple simulations
        """
        if parameter_bounds is None:
            parameter_bounds = {
                "humidity": (0.3, 0.7),
                "light_exposure": (0.2, 0.8),
                "air_pollutants": (0.1, 0.5),
                "temperature": (288, 308)  # 15°C to 35°C
            }

        results = []
        condition_scores = []
        color_changes = []
        darkening_values = []
        yellowing_values = []
        fading_values = []
        cracking_values = []
        
        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().numpy()
        else:
            indices = np.array(pigment_indices)
            
        if mixing_ratios is not None:
            if isinstance(mixing_ratios, torch.Tensor):
                ratios = mixing_ratios.detach().cpu().numpy()
            else:
                ratios = np.array(mixing_ratios)
        else:
            ratios = np.ones(len(indices)) / len(indices)
        
        for i in range(n_samples):
            env_conditions = {}
            for param, bounds in parameter_bounds.items():
                env_conditions[param] = np.random.uniform(bounds[0], bounds[1])

            temperature = env_conditions.get("temperature", 298)

            if isinstance(indices, np.ndarray):
                sim_indices = []
                for idx in indices:
                    sim_indices.append(self._extract_scalar(idx))
            else:
                sim_indices = indices.copy()

            if ratios is not None:
                if isinstance(ratios, np.ndarray):
                    sim_ratios = ratios.copy()
                else:
                    sim_ratios = ratios
            else:
                sim_ratios = None

            result = self.simulate_aging_effects(
                sim_indices, 
                sim_ratios, 
                years, 
                env_conditions,
                temperature
            )

            for j, pigment_id in enumerate(indices):
                if hasattr(pigment_id, 'item'):
                    if hasattr(pigment_id, 'size') and pigment_id.size == 1:
                        pigment_id = pigment_id.item()
                    elif hasattr(pigment_id, 'shape') and len(pigment_id.shape) == 0:
                        pigment_id = pigment_id.item()
                    elif hasattr(pigment_id, 'numel') and pigment_id.numel() == 1:
                        pigment_id = pigment_id.item()
                    elif hasattr(pigment_id, 'size') and pigment_id.size > 1:
                        pigment_id = pigment_id[0].item()
                    elif hasattr(pigment_id, 'shape') and len(pigment_id.shape) > 0:
                        pigment_id = pigment_id[0].item()
                
                try:
                    aging_factors = self.pigment_db.get_aging_factors(pigment_id)
                    if isinstance(aging_factors, dict):
                        for factor in aging_factors:
                            if isinstance(aging_factors[factor], dict):
                                for param in aging_factors[factor]:
                                    aging_factors[factor][param] *= np.random.uniform(0.8, 1.2)
                            else:
                                aging_factors[factor] *= np.random.uniform(0.8, 1.2)
                except:
                    pass
                    
            temperature = env_conditions.get("temperature", 298)
            result = self.simulate_aging_effects(
                indices, 
                ratios, 
                years, 
                env_conditions,
                temperature
            )

            results.append(result)

            condition_scores.append(result["condition_score"])
            color_changes.append(result["color_change"]["delta_e"])
            darkening_values.append(result["aging_effects"]["darkening"])
            yellowing_values.append(result["aging_effects"]["yellowing"])
            fading_values.append(result["aging_effects"]["fading"])
            cracking_values.append(result["aging_effects"]["cracking"])

        stats = self._analyze_monte_carlo_results(
            results, 
            condition_scores, 
            color_changes,
            darkening_values,
            yellowing_values,
            fading_values, 
            cracking_values
        )

        baseline = self.simulate_aging_effects(
            indices,
            ratios,
            years,
            {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3
            }
        )
        stats["baseline_result"] = baseline
        
        return stats

    def _analyze_monte_carlo_results(self, results, condition_scores, color_changes, 
                                    darkening_values, yellowing_values, fading_values, cracking_values):
        """
        Analyze statistical distributions from Monte Carlo simulations.
        
        Args:
            results: List of simulation results
            condition_scores, color_changes, etc.: Lists of specific metrics
            
        Returns:
            Dictionary with statistical analysis
        """
        condition_scores = np.array(condition_scores)
        color_changes = np.array(color_changes)
        darkening_values = np.array(darkening_values)
        yellowing_values = np.array(yellowing_values)
        fading_values = np.array(fading_values)
        cracking_values = np.array(cracking_values)

        stats = {
            "condition_score": {
                "mean": np.mean(condition_scores),
                "std": np.std(condition_scores),
                "min": np.min(condition_scores),
                "max": np.max(condition_scores),
                "q05": np.percentile(condition_scores, 5),
                "q25": np.percentile(condition_scores, 25),
                "q50": np.percentile(condition_scores, 50), 
                "q75": np.percentile(condition_scores, 75),
                "q95": np.percentile(condition_scores, 95)
            },
            "color_change_delta_e": {
                "mean": np.mean(color_changes),
                "std": np.std(color_changes),
                "min": np.min(color_changes),
                "max": np.max(color_changes),
                "q05": np.percentile(color_changes, 5),
                "q50": np.percentile(color_changes, 50),
                "q95": np.percentile(color_changes, 95)
            },
            "aging_effects": {
                "darkening": {
                    "mean": np.mean(darkening_values),
                    "std": np.std(darkening_values),
                    "q05": np.percentile(darkening_values, 5),
                    "q50": np.percentile(darkening_values, 50),
                    "q95": np.percentile(darkening_values, 95)
                },
                "yellowing": {
                    "mean": np.mean(yellowing_values),
                    "std": np.std(yellowing_values),
                    "q05": np.percentile(yellowing_values, 5),
                    "q50": np.percentile(yellowing_values, 50),
                    "q95": np.percentile(yellowing_values, 95)
                },
                "fading": {
                    "mean": np.mean(fading_values),
                    "std": np.std(fading_values),
                    "q05": np.percentile(fading_values, 5),
                    "q50": np.percentile(fading_values, 50),
                    "q95": np.percentile(fading_values, 95)
                },
                "cracking": {
                    "mean": np.mean(cracking_values),
                    "std": np.std(cracking_values),
                    "q05": np.percentile(cracking_values, 5),
                    "q50": np.percentile(cracking_values, 50),
                    "q95": np.percentile(cracking_values, 95)
                }
            },
            "sample_count": len(results),
            "confidence_interval_95": {
                "condition_score": (
                    np.percentile(condition_scores, 2.5),
                    np.percentile(condition_scores, 97.5)
                ),
                "color_change_delta_e": (
                    np.percentile(color_changes, 2.5),
                    np.percentile(color_changes, 97.5)
                )
            },
            "sensitivity": self._calculate_parameter_sensitivity(results)
        }

        mechanism_counts = {}
        for result in results:
            primary = result.get("primary_effect", "unknown")
            if primary not in mechanism_counts:
                mechanism_counts[primary] = 0
            mechanism_counts[primary] += 1

        for mech in mechanism_counts:
            mechanism_counts[mech] = (mechanism_counts[mech] / len(results)) * 100
        
        stats["degradation_mechanisms"] = mechanism_counts

        rating_counts = {"Excellent": 0, "Good": 0, "Moderate": 0, "Poor": 0, "Very Poor": 0}
        for result in results:
            rating = result.get("condition_rating", "unknown")
            if rating in rating_counts:
                rating_counts[rating] += 1

        for rating in rating_counts:
            rating_counts[rating] = (rating_counts[rating] / len(results)) * 100
        
        stats["condition_rating_distribution"] = rating_counts
        
        return stats

    def _calculate_parameter_sensitivity(self, results):
        """
        Calculate parameter sensitivity from Monte Carlo results.
        
        Args:
            results: List of simulation results
            
        Returns:
            Dictionary with sensitivity analysis
        """
        env_params = {
            "humidity": [],
            "light_exposure": [],
            "air_pollutants": [],
            "temperature_fluctuation": []
        }
        
        condition_scores = []
        
        for result in results:
            if hasattr(result, "get") and result.get("environmental_conditions"):
                for param in env_params:
                    if param in result["environmental_conditions"]:
                        env_params[param].append(result["environmental_conditions"][param])

            if hasattr(result, "get") and result.get("condition_score"):
                condition_scores.append(result["condition_score"])

        sensitivity = {}
        for param in env_params:
            if len(env_params[param]) == len(condition_scores) and len(env_params[param]) > 0:
                correlation = np.corrcoef(env_params[param], condition_scores)[0, 1]
                sensitivity[param] = correlation

        sorted_sensitivity = sorted(
            sensitivity.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return {
            "correlations": sensitivity,
            "ranked_impacts": [
                {"parameter": param, "correlation": corr} 
                for param, corr in sorted_sensitivity
            ]
        }

    def generate_spectral_uncertainty_bands(self, pigment_indices, mixing_ratios=None, 
                                        years=100, n_samples=50, spectral_bands=31):
        """
        Generate spectral uncertainty bands for visualization.
        
        Args:
            pigment_indices: Pigment indices
            mixing_ratios: Optional mixing ratios
            years: Simulation time in years
            n_samples: Number of Monte Carlo samples
            spectral_bands: Number of spectral bands
            
        Returns:
            Dictionary with spectral data and uncertainty bands
        """
        mc_results = self.monte_carlo_aging_simulation(
            pigment_indices, 
            mixing_ratios,
            years,
            n_samples
        )

        wavelengths = np.linspace(400, 700, spectral_bands)

        all_aged_spectra = []

        for i in range(n_samples):
            env_conditions = {
                "humidity": np.random.uniform(0.3, 0.7),
                "light_exposure": np.random.uniform(0.2, 0.8),
                "air_pollutants": np.random.uniform(0.1, 0.5)
            }

            aging_data = self.generate_spectral_aging_data(
                pigment_indices,
                mixing_ratios,
                years,
                spectral_bands,
                env_conditions
            )
            
            if aging_data and 'aged_spectral' in aging_data:
                all_aged_spectra.append(aging_data['aged_spectral'])

        if all_aged_spectra:
            spectra_array = np.array(all_aged_spectra)
            mean_spectrum = np.mean(spectra_array, axis=0)
            median_spectrum = np.median(spectra_array, axis=0)
            lower_bound = np.percentile(spectra_array, 5, axis=0)
            upper_bound = np.percentile(spectra_array, 95, axis=0)
            baseline_data = self.generate_spectral_aging_data(
                pigment_indices,
                mixing_ratios,
                years,
                spectral_bands,
                {"humidity": 0.5, "light_exposure": 0.5, "air_pollutants": 0.3}
            )
            
            original_spectral = baseline_data.get('original_spectral', mean_spectrum)
            
            return {
                'wavelengths': wavelengths.tolist(),
                'original_spectral': original_spectral,
                'mean_aged_spectral': mean_spectrum.tolist(),
                'median_aged_spectral': median_spectrum.tolist(),
                'lower_bound_spectral': lower_bound.tolist(),
                'upper_bound_spectral': upper_bound.tolist(),
                'confidence_interval': '90%',
                'sample_count': len(all_aged_spectra),
                'summary_stats': mc_results
            }
        else:
            return None

    def _check_interlayer_interactions(self, bottom_layer, top_layer, bottom_binder, top_binder):
        """Check for issues between adjacent layers."""
        issues = []
        temperature = 298

        for p1 in bottom_layer:
            for p2 in top_layer:
                rate, reaction_type = self._calculate_reaction_rate(p1, p2, temperature)
                if rate > 0:
                    migration_factor = self._calculate_migration_potential(p1, p2, bottom_binder, top_binder)
                    if migration_factor > 0.3:
                        p1_name = self.pigment_db.id_to_name.get(p1, f"Pigment {p1}")
                        p2_name = self.pigment_db.id_to_name.get(p2, f"Pigment {p2}")
                        severity = min(0.9, rate / 1e3) * migration_factor
                        
                        issues.append({
                            "type": "layer_interaction",
                            "pigments": [p1_name, p2_name],
                            "mechanism": reaction_type,
                            "severity": severity,
                            "description": f"Migration and reaction: {self._get_reaction_product(p1, p2)}"
                        })

        if bottom_binder != top_binder:
            problematic_combinations = [
                ("animal_glue", "drying_oil"),
                ("gum_arabic", "drying_oil")
            ]
            
            if (bottom_binder, top_binder) in problematic_combinations:
                issues.append({
                    "type": "binder_interaction",
                    "binders": [bottom_binder, top_binder],
                    "severity": 0.6,
                    "description": "Poor adhesion between these binder types"
                })

            if (bottom_binder in ["drying_oil", "egg_tempera"] and 
                top_binder in ["animal_glue", "gum_arabic"]):
                issues.append({
                    "type": "binder_interaction",
                    "binders": [bottom_binder, top_binder],
                    "severity": 0.5,
                    "description": "Fast-drying layer over slow-drying layer can cause cracking"
                })

        bleeding_pigments = [30, 31, 32] 
        for p in bleeding_pigments:
            if p in bottom_layer and any(self._is_porous_pigment(tp) for tp in top_layer):
                p_name = self.pigment_db.id_to_name.get(p, f"Pigment {p}")
                issues.append({
                    "type": "layer_interaction",
                    "pigments": [p_name],
                    "mechanism": "diffusion",
                    "severity": 0.4,
                    "description": f"Bleeding risk: {p_name} may migrate into upper layer"
                })
                
        return issues
    
    def _calculate_migration_potential(self, pigment1, pigment2, binder1, binder2):
        """Calculate the potential for material migration between layers."""
        
        # 1. Estimate solubility factors based on pigment chemistry
        solubility_factor = 0.5 
        water_soluble = [30, 31, 32, 33] 
        oil_soluble = [3, 14, 10, 11] 
        
        if pigment1 in water_soluble or pigment2 in water_soluble:
            if binder1 in ["animal_glue", "gum_arabic"] or binder2 in ["animal_glue", "gum_arabic"]:
                solubility_factor = 0.8  
                
        if pigment1 in oil_soluble or pigment2 in oil_soluble:
            if binder1 in ["drying_oil", "egg_tempera"] or binder2 in ["drying_oil", "egg_tempera"]:
                solubility_factor = 0.7  
        
        # 2. Binder porosity factors
        porosity_factor = 0.5  
        
        binder_porosity = {
            "animal_glue": 0.8, 
            "gum_arabic": 0.6, 
            "egg_tempera": 0.4,
            "drying_oil": 0.3
        }
        
        combined_porosity = (binder_porosity.get(binder1, 0.5) + 
                            binder_porosity.get(binder2, 0.5)) / 2
        porosity_factor = combined_porosity
        
        # 3. Particle size factor
        size_factor = 0.5 
        
        if (pigment1 in self.nanoscale_properties and 
            pigment2 in self.nanoscale_properties):
            
            p1_size = self.nanoscale_properties[pigment1]["particle_size"]
            p2_size = self.nanoscale_properties[pigment2]["particle_size"]
            
            smallest_size = min(p1_size, p2_size)
            
            if smallest_size < 1.0:
                size_factor = 0.9  
            elif smallest_size < 3.0:
                size_factor = 0.7 
            elif smallest_size < 8.0:
                size_factor = 0.5 
            else:
                size_factor = 0.3
        
        migration_potential = (
            0.4 * solubility_factor +
            0.4 * porosity_factor +
            0.2 * size_factor
        )
        
        return migration_potential
    
    def _is_porous_pigment(self, pigment_id):
        """Check if a pigment tends to create porous paint layers."""
        porous_pigments = [0, 1, 2, 18, 25]  # Kaolin, calcium carbonate, gypsum, lapis, carbon black
        return pigment_id in porous_pigments
    

    def _simulate_interlayer_diffusion(self, layer_pigments, layer_binders, years, environmental_conditions=None):
        """
        Simulate diffusion of components between adjacent paint layers over time.
        
        Args:
            layer_pigments: List of lists of pigment indices for each layer
            layer_binders: List of binder types for each layer
            years: Simulation time in years
            environmental_conditions: Optional dict with environmental factors
            
        Returns:
            List of diffusion results between adjacent layers
        """
        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "temperature": 298,
                "temperature_fluctuation": 0.2
            }

        diffusion_results = []
        for i in range(len(layer_pigments) - 1):
            bottom_layer = layer_pigments[i]
            top_layer = layer_pigments[i+1]
            bottom_binder = layer_binders[i]
            top_binder = layer_binders[i+1]

            migration_factor = self._calculate_migration_potential(
                bottom_layer[0] if isinstance(bottom_layer, list) else bottom_layer,
                top_layer[0] if isinstance(top_layer, list) else top_layer,
                bottom_binder,
                top_binder
            )

            temp = environmental_conditions.get("temperature", 298)
            humidity = environmental_conditions.get("humidity", 0.5)

            diffusion_coef = 1e-6 * np.exp(-40000 / (8.314 * temp))
            diffusion_coef *= (1 + humidity * 2) 

            time_seconds = years * 365 * 24 * 3600
            diffusion_distance = np.sqrt(2 * diffusion_coef * time_seconds)
 
            effective_distance = diffusion_distance * migration_factor

            layer_thickness = 30e-6 
            diffusion_amount = min(1.0, effective_distance / layer_thickness)

            diffusing_pigments = self._identify_diffusing_pigments(
                bottom_layer, top_layer, bottom_binder, top_binder
            )
 
            reaction_potential = self._calculate_interlayer_reaction_potential(
                bottom_layer, top_layer, diffusing_pigments, diffusion_amount
            )

            diffusion_results.append({
                "bottom_layer_index": i,
                "top_layer_index": i+1,
                "migration_factor": migration_factor,
                "diffusion_coefficient": diffusion_coef,
                "diffusion_distance_microns": effective_distance * 1e6,  # Convert to microns
                "diffusion_amount": diffusion_amount,
                "diffusing_pigments": diffusing_pigments,
                "reaction_potential": reaction_potential,
                "stability_impact": self._calculate_diffusion_stability_impact(
                    diffusion_amount, reaction_potential, diffusing_pigments
                )
            })
        
        return diffusion_results

    def _identify_diffusing_pigments(self, bottom_layer, top_layer, bottom_binder, top_binder):
        """
        Identify which pigments are likely to diffuse between layers.
        
        Args:
            bottom_layer, top_layer: Pigment indices for each layer
            bottom_binder, top_binder: Binder types for each layer
            
        Returns:
            List of diffusing pigment information
        """
        diffusing_pigments = []

        if not isinstance(bottom_layer, list):
            bottom_layer = [bottom_layer]
        if not isinstance(top_layer, list):
            top_layer = [top_layer]

        for pigment_id in bottom_layer:
            if hasattr(pigment_id, 'item'):
                pigment_id = pigment_id.item()

            diffusion_likelihood = 0.0
            
            # 1. Check particle size
            try:
                particle_data = self.pigment_db.get_particle_data(pigment_id)
                size = particle_data["size"]["mean"]

                if size < 1.0:
                    diffusion_likelihood += 0.5
                elif size < 3.0:
                    diffusion_likelihood += 0.3
                elif size < 5.0:
                    diffusion_likelihood += 0.1
            except:
                diffusion_likelihood += 0.2
            
            # 2. Check solubility in binder
            if pigment_id in [30, 31, 32, 33]: 
                if top_binder in ["animal_glue", "gum_arabic"]:
                    diffusion_likelihood += 0.4 
                elif top_binder in ["egg_tempera"]:
                    diffusion_likelihood += 0.2
            elif pigment_id in [10, 11, 12, 13, 23]:
                if top_binder in ["drying_oil"]:
                    diffusion_likelihood += 0.1
                    
            # 3. Only record pigments with significant diffusion likelihood
            if diffusion_likelihood > 0.2:
                pigment_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
                diffusing_pigments.append({
                    "id": pigment_id,
                    "name": pigment_name,
                    "direction": "bottom_to_top",
                    "likelihood": diffusion_likelihood,
                    "mechanism": self._get_diffusion_mechanism(pigment_id, bottom_binder, top_binder)
                })

        for pigment_id in top_layer:
            if hasattr(pigment_id, 'item'):
                pigment_id = pigment_id.item()

            diffusion_likelihood = 0.0

            if bottom_binder in ["animal_glue", "gum_arabic"] and top_binder in ["drying_oil"]:
                if pigment_id in [3, 14, 17, 21, 22]: 
                    diffusion_likelihood += 0.2

            if diffusion_likelihood > 0.15:
                pigment_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
                diffusing_pigments.append({
                    "id": pigment_id,
                    "name": pigment_name,
                    "direction": "top_to_bottom",
                    "likelihood": diffusion_likelihood,
                    "mechanism": self._get_diffusion_mechanism(pigment_id, top_binder, bottom_binder)
                })
        
        return diffusing_pigments

    def _get_diffusion_mechanism(self, pigment_id, source_binder, target_binder):
        """Determine the diffusion mechanism based on pigment and binders."""
        if source_binder in ["animal_glue", "gum_arabic"] and target_binder in ["drying_oil"]:
            return "Pigment carried by oil penetration into porous ground"
        elif source_binder in ["drying_oil"] and target_binder in ["animal_glue", "gum_arabic"]:
            return "Migration during binder setting phase"
        elif pigment_id in [30, 31, 32, 33]: 
            return "Soluble organic components migrating between layers"
        else:
            return "Particle migration through porous matrix"

    def _calculate_interlayer_reaction_potential(self, bottom_layer, top_layer, diffusing_pigments, diffusion_amount):
        """
        Calculate the potential for chemical reactions between diffusing pigments.
        
        Args:
            bottom_layer, top_layer: Pigment indices for each layer
            diffusing_pigments: Information about pigments that can diffuse
            diffusion_amount: Amount of diffusion (0-1 scale)
            
        Returns:
            Dictionary with reaction potential information
        """
        if not diffusing_pigments:
            return {"potential": 0.0, "reactions": []}

        bottom_pigments = [p.item() if hasattr(p, 'item') else p for p in bottom_layer]
        top_pigments = [p.item() if hasattr(p, 'item') else p for p in top_layer]
        reactions = []
        total_potential = 0.0
        
        for diff_pig in diffusing_pigments:
            pigment_id = diff_pig["id"]
            direction = diff_pig["direction"]
            target_pigments = top_pigments if direction == "bottom_to_top" else bottom_pigments
            
            for target_id in target_pigments:
                rate, reaction_type = self._calculate_reaction_rate(pigment_id, target_id)
                
                if rate > 0:
                    effective_rate = rate * diffusion_amount * diff_pig["likelihood"]
                    reactions.append({
                        "diffusing_pigment": diff_pig["name"],
                        "target_pigment": self.pigment_db.id_to_name.get(target_id, f"Pigment {target_id}"),
                        "rate": rate,
                        "effective_rate": effective_rate,
                        "type": reaction_type,
                        "product": self._get_reaction_product(pigment_id, target_id)
                    })
                    
                    total_potential += effective_rate

        if reactions:
            total_potential = min(1.0, total_potential / (len(reactions) * 1000))
        
        return {
            "potential": total_potential,
            "reactions": reactions
        }

    def _calculate_diffusion_stability_impact(self, diffusion_amount, reaction_potential, diffusing_pigments):
        """
        Calculate the overall stability impact of interlayer diffusion.
        
        Args:
            diffusion_amount: Amount of diffusion (0-1 scale)
            reaction_potential: Information about potential reactions
            diffusing_pigments: Information about diffusing pigments
            
        Returns:
            Stability impact (0-1 scale) and impact description
        """
        base_impact = diffusion_amount * 0.3
        reaction_impact = reaction_potential["potential"] * 0.7
        total_impact = base_impact + reaction_impact

        if total_impact < 0.1:
            description = "Minimal interlayer interaction"
        elif total_impact < 0.3:
            description = "Minor pigment migration between layers"
        elif total_impact < 0.6:
            description = "Significant interlayer diffusion with some reactions"
        else:
            description = "Severe interlayer migration and chemical interactions"
        
        return {
            "impact_score": total_impact,
            "description": description
        }
    
    def suggest_corrections(self, pigment_indices, mixing_ratios, 
                          binder_type=None, environmental_conditions=None):
        """
        Suggest corrections to improve stability of a pigment mixture.
        
        Args:
            pigment_indices: Tensor/array of pigment indices
            mixing_ratios: Tensor/array of mixing ratios
            binder_type: Optional binder material
            environmental_conditions: Optional dict with environmental factors
            
        Returns:
            Dictionary with corrected ratios and detailed suggestions
        """
        stability_info = self.check_mixture_stability(
            pigment_indices, mixing_ratios, binder_type, environmental_conditions
        )
        if stability_info["is_stable"]:
            return {
                "corrected_ratios": mixing_ratios,
                "original_stability": stability_info,
                "suggestions": [{
                    "type": "confirmation",
                    "message": "Mixture is already stable. No corrections needed."
                }]
            }
        
        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().numpy()
        else:
            indices = np.array(pigment_indices)
            
        if isinstance(mixing_ratios, torch.Tensor):
            ratios = mixing_ratios.detach().cpu().numpy().copy()
        else:
            ratios = np.array(mixing_ratios).copy()

        detailed_issues = stability_info["detailed_issues"]

        problematic_pigments = {}
        for issue in detailed_issues:
            if issue["type"] == "chemical_reaction":
                for pigment in issue["pigments"]:
                    if pigment not in problematic_pigments:
                        problematic_pigments[pigment] = 0
                    problematic_pigments[pigment] += issue["impact"]
            elif issue["type"] == "photodegradation":
                pigment = issue["pigment"]
                if pigment not in problematic_pigments:
                    problematic_pigments[pigment] = 0
                problematic_pigments[pigment] += issue["impact"]
            elif issue["type"] == "nanoscale_interaction":
                for pigment in issue["pigments"]:
                    if pigment not in problematic_pigments:
                        problematic_pigments[pigment] = 0
                    problematic_pigments[pigment] += issue["impact"] * 0.5
        
        suggestions = []
        ratios_modified = False
        if problematic_pigments:
            prob_indices = []
            for pigment_name in problematic_pigments:
                for i, idx in enumerate(indices):
                    if pigment_name == self.pigment_db.id_to_name.get(idx, f"Pigment {idx}"):
                        prob_indices.append((i, problematic_pigments[pigment_name]))
            prob_indices.sort(key=lambda x: x[1], reverse=True)
            for index, impact in prob_indices:
                if ratios[index] < 0.1:
                    continue

                reduction_factor = min(0.7, impact * 2)

                original_ratio = ratios[index]
                ratios[index] *= (1.0 - reduction_factor)
                
                pigment_name = self.pigment_db.id_to_name.get(indices[index], f"Pigment {indices[index]}")
                suggestions.append({
                    "type": "ratio_adjustment",
                    "pigment": pigment_name,
                    "original_ratio": original_ratio,
                    "new_ratio": ratios[index],
                    "reason": f"Reduced concentration to minimize stability issues"
                })
                ratios_modified = True
                pigment_to_replace = indices[index]
                alternatives = self._suggest_alternative_pigments(pigment_to_replace, indices, binder_type)
                
                if alternatives:
                    suggestions.append({
                        "type": "pigment_replacement",
                        "problematic_pigment": pigment_name,
                        "alternatives": alternatives,
                        "reason": f"Consider replacing with a more compatible alternative"
                    })
        
        # 1. Photo-degradation issues
        photo_issues = [issue for issue in detailed_issues if issue["type"] == "photodegradation"]
        if photo_issues and environmental_conditions and "light_exposure" in environmental_conditions:
            suggestions.append({
                "type": "environmental_control",
                "factor": "light_exposure",
                "current_value": environmental_conditions["light_exposure"],
                "recommendation": "Reduce light exposure by using UV filters and controlled lighting",
                "reason": f"Light-sensitive pigments detected: {', '.join(issue['pigment'] for issue in photo_issues)}"
            })
            
        # 2. Microbial risk issues
        microbial_issues = [issue for issue in detailed_issues if issue["type"] == "microbial_risk"]
        if microbial_issues and environmental_conditions and "humidity" in environmental_conditions:
            suggestions.append({
                "type": "environmental_control",
                "factor": "humidity",
                "current_value": environmental_conditions.get("humidity", "unknown"),
                "recommendation": "Maintain humidity below 60% to inhibit microbial growth",
                "reason": "Mixture contains components susceptible to microbiological degradation"
            })
            suggestions.append({
                "type": "additive",
                "additive": "biocide",
                "recommended": True,
                "reason": "A small amount of fungicide/biocide additive would improve long-term stability"
            })
            
        # 3. Binder incompatibility
        binder_issues = [issue for issue in detailed_issues if issue["type"] == "binder_incompatibility"]
        if binder_issues and binder_type:
            alternative_binders = self._suggest_alternative_binders(indices, binder_type)
            if alternative_binders:
                suggestions.append({
                    "type": "binder_change",
                    "current_binder": binder_type,
                    "suggested_binders": alternative_binders,
                    "reason": "More compatible binder for these pigments"
                })
                
        # 4. If spectral instability issues, suggest pigment balance adjustment
        spectral_issues = [issue for issue in detailed_issues if issue["type"] == "spectral_instability"]
        if spectral_issues:
            suggestions.append({
                "type": "color_stability",
                "recommendation": "Consider a protective varnish layer to minimize color shift",
                "reason": "Predicted color instability can be partially mitigated with oxygen barrier"
            })

        if ratios_modified:
            ratios = ratios / np.sum(ratios)
            corrected_stability = self.check_mixture_stability(
                indices, ratios, binder_type, environmental_conditions
            )
            
            if corrected_stability["stability_score"] > stability_info["stability_score"]:
                improvement = corrected_stability["stability_score"] - stability_info["stability_score"]
                suggestions.append({
                    "type": "evaluation",
                    "message": f"Stability score improved by {improvement:.2f} to {corrected_stability['stability_score']:.2f}",
                    "new_stability_rating": corrected_stability["stability_rating"]
                })
            else:
                suggestions.append({
                    "type": "evaluation",
                    "message": "Proposed adjustments have limited effect. Consider replacement options.",
                    "new_stability_rating": corrected_stability["stability_rating"]
                })
        else:
            corrected_stability = stability_info
            
        return {
            "corrected_ratios": ratios,
            "original_stability": stability_info,
            "corrected_stability": corrected_stability,
            "suggestions": suggestions
        }
        
    def _suggest_alternative_pigments(self, pigment_id, current_mixture, binder_type=None):
        """
        Suggest alternative pigments.
        
        Args:
            pigment_id: ID of the pigment to replace
            current_mixture: Current mixture of pigments
            binder_type: Optional binder material
            
        Returns:
            List of alternative pigments with evaluations
        """
        pigment_color = None
        for p in self.pigment_db.pigments:
            if self.pigment_db.name_to_id.get(p["name"]) == pigment_id:
                pigment_color = p["color"]
                break
        
        if not pigment_color:
            return []
        same_color_pigments = self.pigment_db.get_pigments_by_color(pigment_color)
        alternatives = [p for p in same_color_pigments if p not in current_mixture and p != pigment_id]
        stable_alternatives = []
        for alt in alternatives:
            test_mixture = [p for p in current_mixture if p != pigment_id] + [alt]
            test_ratios = np.ones(len(test_mixture)) / len(test_mixture)

            stability = self.check_mixture_stability(test_mixture, test_ratios, binder_type)
            
            if stability["stability_score"] > 0.5:
                alt_name = "Unknown"
                chemical_formula = ""
                
                for p in self.pigment_db.pigments:
                    if self.pigment_db.name_to_id.get(p["name"]) == alt:
                        alt_name = p["name"]
                        chemical_formula = p.get("formula", "")
                        break

                spectral_similarity = self._calculate_spectral_similarity(pigment_id, alt)

                stable_alternatives.append({
                    "id": alt,
                    "name": alt_name,
                    "formula": chemical_formula,
                    "stability_score": stability["stability_score"],
                    "stability_rating": stability["stability_rating"],
                    "spectral_similarity": spectral_similarity,
                    "historically_accurate": self._is_historically_appropriate(alt)
                })

        stable_alternatives.sort(
            key=lambda x: (x["stability_score"] * 0.7 + x["spectral_similarity"] * 0.3), 
            reverse=True
        )
        
        return stable_alternatives
    
    def _calculate_spectral_similarity(self, pigment1, pigment2):
        """Calculate spectral similarity between two pigments."""
        if pigment1 not in self.spectral_data or pigment2 not in self.spectral_data:
            return 0.5  
            
        spec1 = self.spectral_data[pigment1]['reflectance']
        spec2 = self.spectral_data[pigment2]['reflectance']
        correlation = np.corrcoef(spec1, spec2)[0, 1]
        if np.isnan(correlation):
            correlation = 0.5
        similarity = (correlation + 1) / 2
        
        return similarity
    
    def _suggest_alternative_binders(self, pigment_indices, current_binder):
        """Suggest alternative binders with better compatibility."""
        if current_binder not in self.binder_compatibility:
            return []
            
        all_binders = list(self.binder_compatibility.keys())
        alternative_binders = [b for b in all_binders if b != current_binder]

        binder_scores = []
        for binder in alternative_binders:
            total_score = 0
            for pigment_id in pigment_indices:
                if pigment_id in self.binder_compatibility[binder]:
                    total_score += self.binder_compatibility[binder][pigment_id]
                else:
                    total_score += 0.5  
            
            avg_score = total_score / len(pigment_indices)
            current_avg = 0
            for pigment_id in pigment_indices:
                if pigment_id in self.binder_compatibility[current_binder]:
                    current_avg += self.binder_compatibility[current_binder][pigment_id]
                else:
                    current_avg += 0.5
            current_avg /= len(pigment_indices)
            
            if avg_score > current_avg:
                binder_scores.append({
                    "binder": binder,
                    "compatibility_score": avg_score,
                    "improvement": avg_score - current_avg,
                    "historically_accurate": self._is_historically_appropriate_binder(binder)
                })

        binder_scores.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        return binder_scores
    
    def _is_historically_appropriate(self, pigment_id):
        """Check if pigment was used in historical Dunhuang pigment practice."""
        for period, data in self.historical_data.items():
            for group in data["compatible_groups"]:
                if pigment_id in group:
                    return True
        return False
    
    def _is_historically_appropriate_binder(self, binder_type):
        """Check if binder was used in historical practice."""
        historical_binders = ["animal_glue", "gum_arabic"]
        return binder_type in historical_binders
    
    def simulate_aging_effects(self, pigment_indices, mixing_ratios, 
                         years=100, environmental_conditions=None,
                         temperature=298):
        """
        Simulate long-term aging effects.
        
        Args:
            pigment_indices: Array of pigment indices
            mixing_ratios: Array of mixing ratios
            years: Simulation time in years
            environmental_conditions: Optional dict with environmental factors
            temperature: Temperature in K (default 298K = 25°C)
            
        Returns:
            Dictionary with comprehensive aging simulation results
        """
        indices = self._preprocess_indices(pigment_indices)

        if mixing_ratios is None:
            ratios = np.ones(len(indices)) / len(indices)
        else:
            if isinstance(mixing_ratios, torch.Tensor):
                ratios = mixing_ratios.detach().cpu().numpy()
            else:
                ratios = np.array(mixing_ratios)

            if np.sum(ratios) > 0:
                ratios = ratios / np.sum(ratios)
            else:
                ratios = np.ones(len(indices)) / len(indices)

        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3
            }
            
        stability_info = self.check_mixture_stability(
            indices, ratios, 
            environmental_conditions=environmental_conditions, 
            temperature=temperature
        )
        stability_score = stability_info['stability_score']
        
        base_instability = max(0, 1.0 - stability_score)
        darkening_total = base_instability * 0.3 
        yellowing_total = base_instability * 0.2 
        fading_total = base_instability * 0.1 
        cracking_total = base_instability * 0.2 

        aging_stages = [
            {"fraction": 0.2, "name": "Initial-settling", "days": 365},        
            {"fraction": 0.4, "name": "Medium-term", "days": 365 * 9},        
            {"fraction": 0.4, "name": "Long-term", "days": 365 * (years - 10)} 
        ]
        if years <= 1:
            aging_stages = [{"fraction": 1.0, "name": "Short-term", "days": 365 * years}]
        elif years <= 10:
            aging_stages = [
                {"fraction": 0.2, "name": "Initial-settling", "days": 365},
                {"fraction": 0.8, "name": "Medium-term", "days": 365 * (years - 1)}
            ]
        
        pigment_specific_effects = {}
        for i, pigment_id in enumerate(indices):
            if isinstance(pigment_id, (list, np.ndarray, torch.Tensor)):
                scalar_pid = self._extract_scalar(pigment_id)
            else:
                scalar_pid = pigment_id

            pigment_name = self.pigment_db.id_to_name.get(scalar_pid, f"Pigment {scalar_pid}")
                
            pigment_specific_effects[pigment_name] = {
                "darkening": darkening_total,  
                "yellowing": yellowing_total,
                "fading": fading_total,
                "cracking": cracking_total,
                "mechanisms": []
            }
            
            for issue in stability_info.get('detailed_issues', []):
                if issue['type'] == 'chemical_reaction' and pigment_name in issue.get('pigments', []):
                    mechanism = issue.get('mechanism', 'chemical_reaction')
                    if mechanism not in pigment_specific_effects[pigment_name]["mechanisms"]:
                        pigment_specific_effects[pigment_name]["mechanisms"].append(mechanism)

        for stage in aging_stages:
            stage_darkening = 0.0
            stage_yellowing = 0.0
            stage_fading = 0.0
            stage_cracking = 0.0
            time_seconds = stage["days"] * 24 * 3600

            for i, pigment_id in enumerate(indices):
                current_ratio = ratios[i]
                if isinstance(current_ratio, np.ndarray):
                    current_ratio = np.mean(current_ratio)
                    
                if current_ratio < 0.05:
                    continue

                try:
                    pigment_id = self._extract_scalar(pigment_id)
                    pigment_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
                except (AttributeError, IndexError):
                    pigment_name = f"Pigment {pigment_id}"

                chemical_interactions = []
                
                for j, other_id in enumerate(indices):
                    if i == j or ratios[j] < 0.05:
                        continue

                    rate, reaction_type = self._calculate_reaction_rate(pigment_id, other_id, temperature)
                    
                    if rate > 0:
                        effective_rate = rate * min(ratios[i], ratios[j]) * (1.0 + years / 100.0)
                        chemical_interactions.append({
                            "partner": other_id,
                            "rate": effective_rate,
                            "type": reaction_type
                        })

                if "light_exposure" in environmental_conditions:
                    light_exposure = environmental_conditions["light_exposure"]
                    for i, pigment_id in enumerate(indices):
                        current_ratio = ratios[i]
                        if isinstance(current_ratio, np.ndarray):
                            current_ratio = np.mean(current_ratio)
                                
                        if current_ratio < 0.05:
                            continue

                        pigment_id = self._extract_scalar(pigment_id)
                        degradation_result = self._calculate_quantum_efficiency_factor(
                            pigment_id, light_exposure
                        )

                    if isinstance(degradation_result, tuple):
                        light_factor, degradation_info = degradation_result
                    else:
                        light_factor = degradation_result
                        degradation_info = {"mechanism": "photodegradation"}

                    if light_factor > 0.1:
                        current_ratio = ratios[i]
                        if isinstance(current_ratio, np.ndarray):
                            current_ratio = np.mean(current_ratio)
                        if pigment_id in [13, 23, 30, 31, 32]:
                            pigment_specific_effects[pigment_name]["fading"] += light_factor * stage["fraction"]
                            stage_fading += light_factor * current_ratio * stage["fraction"]
                        elif pigment_id in [10, 11]: 
                            pigment_specific_effects[pigment_name]["darkening"] += light_factor * stage["fraction"]
                            stage_darkening += light_factor * current_ratio * stage["fraction"]
                
                for interaction in chemical_interactions:
                    other_id = interaction["partner"]
                    
                    try:
                        other_name = self.pigment_db.id_to_name.get(other_id, f"Pigment {other_id}")
                    except (AttributeError, IndexError):
                        other_name = f"Pigment {other_id}"
                    
                    effect_magnitude = 1.0 - np.exp(-interaction["rate"] * time_seconds * 1e-6 * (1 + base_instability))
                    effect_magnitude = min(0.9, effect_magnitude) 
                    
                    is_lead_sulfide = ((pigment_id in [3, 5, 9, 14] and other_id in [10, 11, 13, 23, 24]) or
                                    (other_id in [3, 5, 9, 14] and pigment_id in [10, 11, 13, 23, 24]))
                    
                    is_copper_sulfide = ((pigment_id in [17, 21, 22] and other_id in [10, 11, 13, 23, 24]) or
                                        (other_id in [17, 21, 22] and pigment_id in [10, 11, 13, 23, 24]))
                    
                    if is_lead_sulfide:
                        effect_magnitude *= 2.0 
                    
                    if is_copper_sulfide:
                        effect_magnitude *= 1.5
                    
                    current_ratio = ratios[i]
                    if isinstance(current_ratio, np.ndarray):
                        current_ratio = np.mean(current_ratio)
                    if interaction["type"] == "chemical_reaction":
                        pigment_specific_effects[pigment_name]["darkening"] += effect_magnitude * stage["fraction"]
                        stage_darkening += effect_magnitude * current_ratio * stage["fraction"]

                        mechanism = f"Reaction with {other_name}"
                        if mechanism not in pigment_specific_effects[pigment_name]["mechanisms"]:
                            pigment_specific_effects[pigment_name]["mechanisms"].append(mechanism)
                    
                    elif interaction["type"] == "catalytic_degradation":
                        pigment_specific_effects[pigment_name]["fading"] += effect_magnitude * 0.6 * stage["fraction"]
                        pigment_specific_effects[pigment_name]["yellowing"] += effect_magnitude * 0.4 * stage["fraction"]
                        
                        stage_fading += effect_magnitude * 0.6 * current_ratio * stage["fraction"]
                        stage_yellowing += effect_magnitude * 0.4 * current_ratio * stage["fraction"]

                        mechanism = f"Catalyzed by {other_name}"
                        if mechanism not in pigment_specific_effects[pigment_name]["mechanisms"]:
                            pigment_specific_effects[pigment_name]["mechanisms"].append(mechanism)
                            
                    elif interaction["type"] == "acid_base_reaction":  
                        pigment_specific_effects[pigment_name]["yellowing"] += effect_magnitude * 0.3 * stage["fraction"]
                        pigment_specific_effects[pigment_name]["cracking"] += effect_magnitude * 0.5 * stage["fraction"]
                        
                        stage_yellowing += effect_magnitude * 0.3 * current_ratio * stage["fraction"]
                        stage_cracking += effect_magnitude * 0.5 * current_ratio * stage["fraction"]
                        
                        mechanism = f"pH interaction with {other_name}"
                        if mechanism not in pigment_specific_effects[pigment_name]["mechanisms"]:
                            pigment_specific_effects[pigment_name]["mechanisms"].append(mechanism)
                
                if "humidity" in environmental_conditions and "temperature_fluctuation" in environmental_conditions:
                    humidity = environmental_conditions["humidity"]
                    temp_fluctuation = environmental_conditions["temperature_fluctuation"]
                    
                    env_stress = humidity * 0.5 + temp_fluctuation * 0.5
                    
                    time_years = stage["days"] / 365.0
                    cracking_factor = 0.2 * env_stress * stage["fraction"] * (1.0 - np.exp(-0.1 * time_years))

                    cracking_factor *= (1.0 + base_instability)
                    cracking_factor = min(0.8, cracking_factor) 

                    if pigment_id in [3, 17, 22]:       # Lead white, azurite, malachite
                        cracking_factor *= 1.5  
                    elif pigment_id in [25, 10, 11]:    # Carbon black, cinnabar, vermilion
                        cracking_factor *= 0.7 
                    
                    pigment_specific_effects[pigment_name]["cracking"] += cracking_factor
                    stage_cracking += cracking_factor * ratios[i]
                    
                    if env_stress > 0.6:
                        mechanism = "Humidity and temperature fluctuations"
                        if mechanism not in pigment_specific_effects[pigment_name]["mechanisms"]:
                            pigment_specific_effects[pigment_name]["mechanisms"].append(mechanism)

            darkening_total += stage_darkening
            yellowing_total += stage_yellowing
            fading_total += stage_fading
            cracking_total += stage_cracking
        
        darkening_total = min(1.0, darkening_total)
        yellowing_total = min(1.0, yellowing_total)
        fading_total = min(1.0, fading_total)
        cracking_total = min(1.0, cracking_total)

        if cracking_total > 0.3 and darkening_total > 0.2:
            synergy_factor = cracking_total * darkening_total * 0.3
            darkening_total = min(1.0, darkening_total + synergy_factor)
            
            for name in pigment_specific_effects:
                if pigment_specific_effects[name]["cracking"] > 0.3:
                    pigment_specific_effects[name]["darkening"] = min(
                        1.0, 
                        pigment_specific_effects[name]["darkening"] + synergy_factor
                    )
                    if "Accelerated by cracking" not in pigment_specific_effects[name]["mechanisms"]:
                        pigment_specific_effects[name]["mechanisms"].append("Accelerated by cracking")

        original_rgb, aged_rgb = self._calculate_aged_color(
            indices, ratios, 
            darkening_total, yellowing_total, fading_total
        )

        original_lab = self._rgb_to_lab(original_rgb)
        aged_lab = self._rgb_to_lab(aged_rgb)
        delta_e = self._calculate_ciede2000(original_lab, aged_lab)

        if stability_score < 0.5 and delta_e < 10:
            delta_e = 10 + (0.5 - stability_score) * 40 

        visual_degradation = (darkening_total + yellowing_total + fading_total) / 3
        physical_degradation = cracking_total
  
        condition_score = 1.0 - (0.4 * visual_degradation + 0.3 * physical_degradation + 0.3 * (1.0 - stability_score))
        condition_score = max(0.0, min(1.0, condition_score))

        if condition_score >= 0.8:
            condition_rating = "Excellent"
        elif condition_score >= 0.6:
            condition_rating = "Good"
        elif condition_score >= 0.4:
            condition_rating = "Fair"
        elif condition_score >= 0.2:
            condition_rating = "Poor"
        else:
            condition_rating = "Very Poor"
            
        effects = {
            "darkening": darkening_total,
            "yellowing": yellowing_total,
            "fading": fading_total,
            "cracking": cracking_total
        }
        primary_effect = max(effects, key=effects.get)
    
        explanation = f"After {years} years, this mixture is predicted to exhibit"
        effects_added = False

        if delta_e > 30:
            explanation += f" dramatic color changes (delta_e={delta_e:.1f})"
            effects_added = True
        elif delta_e > 15:
            explanation += f" significant color changes (delta_e={delta_e:.1f})"
            effects_added = True
        elif delta_e > 5:
            explanation += f" noticeable color changes (delta_e={delta_e:.1f})"
            effects_added = True
        elif delta_e > 2:
            explanation += f" subtle color changes (delta_e={delta_e:.1f})"
            effects_added = True

        if not effects_added:
            if darkening_total > 0.3:
                explanation += " significant darkening"
                effects_added = True
            elif darkening_total > 0.1:
                explanation += " moderate darkening"
                effects_added = True
                
            if yellowing_total > 0.3:
                explanation += ", noticeable yellowing" if effects_added else " noticeable yellowing"
                effects_added = True
            elif yellowing_total > 0.1:
                explanation += ", slight yellowing" if effects_added else " slight yellowing"
                effects_added = True
                
            if fading_total > 0.3:
                explanation += ", substantial fading" if effects_added else " substantial fading"
                effects_added = True
            elif fading_total > 0.1:
                explanation += ", gradual fading" if effects_added else " gradual fading"
                effects_added = True
                
            if cracking_total > 0.3:
                explanation += ", and severe cracking" if effects_added else " severe cracking"
                effects_added = True
            elif cracking_total > 0.1:
                explanation += ", and minor cracking" if effects_added else " minor cracking"
                effects_added = True

        if not effects_added:
            explanation += " excellent stability with minimal visible changes"

        explanation += "."

        has_lead = any(self._element_in_list(p, [3, 5, 9, 14]) for p in indices)
        has_sulfur = any(self._element_in_list(p, [10, 11, 13, 23, 24]) for p in indices)
        if has_lead and has_sulfur and darkening_total > 0.2:
            explanation += " Lead compounds will react with sulfur-containing pigments to form black lead sulfide."

        has_copper = any(p in [17, 21, 22] for p in indices)
        if has_copper and has_sulfur and darkening_total > 0.2:
            explanation += " Copper pigments may darken due to reaction with sulfur-containing materials."
  
        recommendations = []
        if stability_score < 0.7:
            recommendations.append("Consider separating these pigments or using a barrier layer between them.")
        if darkening_total > 0.2 or yellowing_total > 0.2:
            recommendations.append("Minimize light exposure to reduce color shifts.")
        if fading_total > 0.2:
            recommendations.append("Consider using UV-filtering glass or display in low light conditions.")
        if cracking_total > 0.2:
            recommendations.append("Maintain stable humidity (45-55%) to minimize cracking.")
        if condition_score < 0.5:
            recommendations.append("This mixture may require significant conservation efforts over time.")

        all_mechanisms = []
        for pigment, effects in pigment_specific_effects.items():
            all_mechanisms.extend(effects["mechanisms"])

        mechanism_counts = {}
        for mech in all_mechanisms:
            if mech not in mechanism_counts:
                mechanism_counts[mech] = 0
            mechanism_counts[mech] += 1
            
        for mech, count in sorted(mechanism_counts.items(), key=lambda x: x[1], reverse=True)[:2]:
            if "Reaction with" in mech and count > 1:
                recommendations.append("Consider separating reactive pigments into different layers.")
            elif "Catalyzed by" in mech:
                recommendations.append("Minimize contact with catalytic materials.")
            elif "UV-" in mech:
                recommendations.append("Use UV-blocking varnish to protect light-sensitive components.")
            elif "pH interaction" in mech:
                recommendations.append("Use pH-neutral materials in framing and conservation.")
                
        return {
            "years_simulated": years,
            "condition_score": condition_score,
            "condition_rating": condition_rating,
            "aging_effects": {
                "darkening": darkening_total,
                "yellowing": yellowing_total,
                "fading": fading_total,
                "cracking": cracking_total
            },
            "primary_effect": primary_effect,
            "pigment_specific_effects": pigment_specific_effects,
            "color_change": {
                "original_rgb": original_rgb,
                "aged_rgb": aged_rgb,
                "delta_e": delta_e
            },
            "explanation": explanation,
            "recommendations": recommendations
        }
    
    def analyze_inter_layer_reactions(self, layer_pigments, layer_binders, environmental_conditions=None):
        """
        Analyze chemical reactions between adjacent paint layers.
        
        Args:
            layer_pigments: List of lists of pigment indices for each layer
            layer_binders: List of binder types for each layer
            environmental_conditions: Optional dict with environmental factors
            
        Returns:
            List of detected interlayer reactions with severity scores
        """
        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "temperature": 298,
                "temperature_fluctuation": 0.3
            }

        reactions = []

        for i in range(len(layer_pigments) - 1):
            upper_layer = layer_pigments[i]
            lower_layer = layer_pigments[i+1]
            upper_binder = layer_binders[i]
            lower_binder = layer_binders[i+1]
            lead_sulfur = self._check_lead_sulfur_reaction(upper_layer, lower_layer)
            if lead_sulfur["detected"]:
                reactions.append({
                    "upper_layer_index": i,
                    "lower_layer_index": i+1,
                    "type": "lead_sulfur_reaction",
                    "severity": lead_sulfur["severity"],
                    "description": lead_sulfur["description"],
                    "affected_pigments": lead_sulfur["pigments"]
                })

            copper_sulfur = self._check_copper_sulfur_reaction(upper_layer, lower_layer)
            if copper_sulfur["detected"]:
                reactions.append({
                    "upper_layer_index": i,
                    "lower_layer_index": i+1,
                    "type": "copper_sulfur_reaction",
                    "severity": copper_sulfur["severity"],
                    "description": copper_sulfur["description"],
                    "affected_pigments": copper_sulfur["pigments"]
                })

            if environmental_conditions["humidity"] > 0.6:
                humid_reactions = self._check_humidity_catalyzed_reactions(
                    upper_layer, lower_layer, 
                    upper_binder, lower_binder, 
                    environmental_conditions["humidity"]
                )
                reactions.extend(humid_reactions)

            acid_base = self._check_acid_base_reactions(
                upper_layer, lower_layer, 
                upper_binder, lower_binder
            )
            if acid_base:
                reactions.extend(acid_base)

            if environmental_conditions.get("temperature_fluctuation", 0) > 0.4:
                temp_reactions = self._check_temperature_catalyzed_reactions(
                    upper_layer, lower_layer, 
                    environmental_conditions["temperature"],
                    environmental_conditions["temperature_fluctuation"]
                )
                reactions.extend(temp_reactions)
        
        return reactions

    def _check_lead_sulfur_reaction(self, upper_layer, lower_layer):
        """Check for lead-sulfur reactions between layers."""
        if not isinstance(upper_layer, list):
            upper_layer = [upper_layer]
        if not isinstance(lower_layer, list):
            lower_layer = [lower_layer]

        upper_layer = [p.item() if hasattr(p, 'item') else p for p in upper_layer]
        lower_layer = [p.item() if hasattr(p, 'item') else p for p in lower_layer]

        lead_pigments = [3, 5, 9, 14]  # lead white, lead chloride, lead sulfate, lead oxide
        sulfur_pigments = [10, 11, 13, 23, 24]  # cinnabar, vermilion, realgar, orpiment, arsenic sulfide

        detected_lead_upper = any(p in lead_pigments for p in upper_layer)
        detected_sulfur_lower = any(p in sulfur_pigments for p in lower_layer)
        detected_lead_lower = any(p in lead_pigments for p in lower_layer)
        detected_sulfur_upper = any(p in sulfur_pigments for p in upper_layer)

        involved_pigments = []
        if detected_lead_upper and detected_sulfur_lower:
            for p in upper_layer:
                if p in lead_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "upper"
                    })
            for p in lower_layer:
                if p in sulfur_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "lower"
                    })
        elif detected_lead_lower and detected_sulfur_upper:
            for p in lower_layer:
                if p in lead_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "lower"
                    })
            for p in upper_layer:
                if p in sulfur_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "upper"
                    })
        
        detected = len(involved_pigments) >= 2
        
        if detected:
            description = "Lead pigments can react with sulfur-containing pigments to form black lead sulfide, causing darkening at the interface between layers."
            lead_count = sum(1 for p in involved_pigments if p["id"] in lead_pigments)
            sulfur_count = sum(1 for p in involved_pigments if p["id"] in sulfur_pigments)
            severity = 0.3 + (0.1 * lead_count) + (0.1 * sulfur_count)
            severity = min(0.9, severity)
        else:
            description = "No lead-sulfur reaction detected between these layers."
            severity = 0.0
            
        return {
            "detected": detected,
            "severity": severity,
            "description": description,
            "pigments": involved_pigments
        }

    def _check_copper_sulfur_reaction(self, upper_layer, lower_layer):
        """Check for copper-sulfur reactions between layers."""
        if not isinstance(upper_layer, list):
            upper_layer = [upper_layer]
        if not isinstance(lower_layer, list):
            lower_layer = [lower_layer]

        upper_layer = [p.item() if hasattr(p, 'item') else p for p in upper_layer]
        lower_layer = [p.item() if hasattr(p, 'item') else p for p in lower_layer]

        copper_pigments = [17, 21, 22]  # azurite, atacamite, malachite
        sulfur_pigments = [10, 11, 13, 23, 24]

        detected_copper_upper = any(p in copper_pigments for p in upper_layer)
        detected_sulfur_lower = any(p in sulfur_pigments for p in lower_layer)
        detected_copper_lower = any(p in copper_pigments for p in lower_layer)
        detected_sulfur_upper = any(p in sulfur_pigments for p in upper_layer)

        involved_pigments = []
        if detected_copper_upper and detected_sulfur_lower:
            for p in upper_layer:
                if p in copper_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "upper"
                    })
            for p in lower_layer:
                if p in sulfur_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "lower"
                    })
        elif detected_copper_lower and detected_sulfur_upper:
            for p in lower_layer:
                if p in copper_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "lower"
                    })
            for p in upper_layer:
                if p in sulfur_pigments:
                    involved_pigments.append({
                        "id": p,
                        "name": self.pigment_db.id_to_name.get(p, f"Pigment {p}"),
                        "layer": "upper"
                    })
        
        detected = len(involved_pigments) >= 2
        
        if detected:
            description = "Copper-based pigments can react with sulfur-containing pigments to form copper sulfide, causing darkening and color shift toward brown/black at layer interfaces."
            
            copper_count = sum(1 for p in involved_pigments if p["id"] in copper_pigments)
            sulfur_count = sum(1 for p in involved_pigments if p["id"] in sulfur_pigments)
            severity = 0.25 + (0.1 * copper_count) + (0.1 * sulfur_count)
            severity = min(0.8, severity)
        else:
            description = "No copper-sulfur reaction detected between these layers."
            severity = 0.0
            
        return {
            "detected": detected,
            "severity": severity,
            "description": description,
            "pigments": involved_pigments
        }

    def _check_humidity_catalyzed_reactions(self, upper_layer, lower_layer, 
                                        upper_binder, lower_binder, humidity):
        """Check for reactions catalyzed by high humidity between layers."""
        reactions = []

        if not isinstance(upper_layer, list):
            upper_layer = [upper_layer]
        if not isinstance(lower_layer, list):
            lower_layer = [lower_layer]

        upper_layer = [p.item() if hasattr(p, 'item') else p for p in upper_layer]
        lower_layer = [p.item() if hasattr(p, 'item') else p for p in lower_layer]

        if humidity > 0.7:
            if upper_binder in ["animal_glue", "gum_arabic"] and any(p in [17, 21, 22] for p in upper_layer):
                reactions.append({
                    "type": "copper_binder_hydrolysis",
                    "upper_layer_index": 0,
                    "lower_layer_index": 1,
                    "severity": humidity * 0.5,
                    "description": "High humidity can cause copper pigments to catalyze degradation of glue or gum binders."
                })

            hygroscopic_pigments = [0, 1, 2, 6, 7] 
            if any(p in hygroscopic_pigments for p in upper_layer) and any(p in hygroscopic_pigments for p in lower_layer):
                reactions.append({
                    "type": "hygroscopic_swelling",
                    "upper_layer_index": 0,
                    "lower_layer_index": 1,
                    "severity": humidity * 0.6,
                    "description": "Hygroscopic pigments in adjacent layers can cause differential swelling, creating mechanical stress at the interface."
                })
        
        return reactions

    def _check_acid_base_reactions(self, upper_layer, lower_layer, upper_binder, lower_binder):
        """Check for acid-base reactions between layers."""
        reactions = []

        if not isinstance(upper_layer, list):
            upper_layer = [upper_layer]
        if not isinstance(lower_layer, list):
            lower_layer = [lower_layer]

        upper_layer = [p.item() if hasattr(p, 'item') else p for p in upper_layer]
        lower_layer = [p.item() if hasattr(p, 'item') else p for p in lower_layer]

        acidic_binders = ["drying_oil"] 
        alkaline_pigments = [0, 1, 2, 3]  # Kaolin, calcium carbonate, gypsum, lead white
        acidic_pigments = [12, 13, 23]  # Hematite, realgar, orpiment

        if upper_binder in acidic_binders and any(p in alkaline_pigments for p in lower_layer):
            alkaline_names = [self.pigment_db.id_to_name.get(p, f"Pigment {p}") 
                            for p in lower_layer if p in alkaline_pigments]
            
            reactions.append({
                "type": "acid_base_reaction",
                "upper_layer_index": 0,
                "lower_layer_index": 1,
                "severity": 0.4,
                "description": f"Acidic {upper_binder} binder can react with alkaline pigments ({', '.join(alkaline_names)}) in the lower layer, causing saponification and potential adhesion issues."
            })

        if any(p in alkaline_pigments for p in upper_layer) and any(p in acidic_pigments for p in lower_layer):
            alkaline_names = [self.pigment_db.id_to_name.get(p, f"Pigment {p}") 
                            for p in upper_layer if p in alkaline_pigments]
            acidic_names = [self.pigment_db.id_to_name.get(p, f"Pigment {p}") 
                        for p in lower_layer if p in acidic_pigments]
            
            reactions.append({
                "type": "acid_base_neutralization",
                "upper_layer_index": 0,
                "lower_layer_index": 1,
                "severity": 0.3,
                "description": f"Alkaline pigments ({', '.join(alkaline_names)}) in the upper layer can neutralize acidic pigments ({', '.join(acidic_names)}) in the lower layer, potentially affecting color."
            })
        
        return reactions

    def _check_temperature_catalyzed_reactions(self, upper_layer, lower_layer, 
                                            temperature, fluctuation):
        """Check for reactions catalyzed by temperature fluctuations."""
        reactions = []

        if not isinstance(upper_layer, list):
            upper_layer = [upper_layer]
        if not isinstance(lower_layer, list):
            lower_layer = [lower_layer]
            
        upper_layer = [p.item() if hasattr(p, 'item') else p for p in upper_layer]
        lower_layer = [p.item() if hasattr(p, 'item') else p for p in lower_layer]
        
        metal_pigments = [3, 5, 9, 14, 17, 21, 22]  # Lead and copper pigments
        organic_pigments = [30, 31, 32, 33]         # Organic pigments
        
        if (any(p in metal_pigments for p in upper_layer) and any(p in organic_pigments for p in lower_layer)) or \
        (any(p in metal_pigments for p in lower_layer) and any(p in organic_pigments for p in upper_layer)):
            temp_factor = np.exp((temperature - 298) / 20)  
            fluct_severity = fluctuation * 0.8
            severity = min(0.8, fluct_severity * temp_factor)
            reactions.append({
                "type": "thermal_catalyzed_degradation",
                "upper_layer_index": 0,
                "lower_layer_index": 1,
                "severity": severity,
                "description": "Temperature fluctuations can accelerate metal-catalyzed degradation of organic components across layer boundaries, as well as create mechanical stress due to differential expansion."
            })
        
        return reactions
    
    def _calculate_aged_color(self, pigment_indices, mixing_ratios, darkening, yellowing, fading):
        """
        Calculate the color change of pigment mixture after aging.
        
        Args:
            pigment_indices: Array of pigment IDs
            mixing_ratios: Array of mixing ratios
            darkening, yellowing, fading: Aging effect magnitudes
            
        Returns:
            Tuple of (original_rgb, aged_rgb)
        """
        mixed_spectrum = np.zeros(31)

        processed_indices = []
        for p in pigment_indices:
            if hasattr(p, 'item'):
                if hasattr(p, 'numel') and p.numel() > 1:
                    processed_indices.append(p[0].item())
                elif hasattr(p, 'size') and p.size > 1:
                    processed_indices.append(p[0].item())
                else:
                    processed_indices.append(p.item())
            else:
                processed_indices.append(p)
            
        pigment_indices = processed_indices

        for i, pigment_id in enumerate(pigment_indices):
            pigment_id = self._extract_scalar(pigment_id)
            if pigment_id in self.spectral_data:
                reflectance = np.array(self.spectral_data[pigment_id]['reflectance'])
                ratio = mixing_ratios[i]
                if isinstance(ratio, np.ndarray):
                    ratio = np.mean(ratio)
                
                weighted_spectrum = reflectance * ratio
                mixed_spectrum += weighted_spectrum

        if np.max(mixed_spectrum) > 1.0:
            mixed_spectrum = mixed_spectrum / np.max(mixed_spectrum)

        aged_spectrum = mixed_spectrum.copy()
        wavelengths = np.linspace(400, 700, 31)
        
        # 1. Darkening
        darkening_factor = darkening * 2.0 
        darkening_mask = 1.0 - (darkening_factor * np.linspace(0.7, 1.3, 31)) 
        darkening_mask = np.clip(darkening_mask, 0.0, 1.0)
        aged_spectrum *= darkening_mask
        
        # 2. Yellowing
        blue_region = wavelengths < 480
        yellow_region = (wavelengths >= 550) & (wavelengths < 600)
        yellowing_factor = yellowing * 2.5
        if np.any(blue_region):
            aged_spectrum[blue_region] *= (1.0 - yellowing_factor * 1.5)
        if np.any(yellow_region):
            aged_spectrum[yellow_region] *= (1.0 + yellowing_factor * 1.2)
        
        # 3. Fading
        if fading > 0.05:
            fade_weights = 0.3 + 0.7 * np.sin(np.linspace(0, np.pi, 31))
            fade_amount = fading * 2.0 * fade_weights
            for i in range(len(aged_spectrum)):
                aged_spectrum[i] = aged_spectrum[i] * (1.0 - fade_amount[i]) + fade_amount[i]
        
        # 4. Add chemical-specific shifts for common pigment interactions
        lead_white_idx = 3
        lead_pigments = [3, 5, 9, 14]
        sulfur_pigments = [10, 11, 13, 23, 24]
        copper_pigments = [17, 21, 22]

        has_lead = any(self._element_in_list(p, lead_pigments) for p in pigment_indices)
        has_sulfur = any(self._element_in_list(p, sulfur_pigments) for p in pigment_indices)
        has_copper = any(self._element_in_list(p, copper_pigments) for p in pigment_indices)
        if has_lead and has_sulfur:
            darkening_strength = max(0.7, 0.7 + darkening * 0.3)
            aged_spectrum *= darkening_strength
        
        # 5. Add copper compound greening from verdigris formation
        copper_pigments = [17, 21, 22]  # Azurite, atacamite, malachite
        has_copper = any(p in copper_pigments for p in pigment_indices)
        if has_copper and has_sulfur:
            green_region = (wavelengths >= 500) & (wavelengths < 550)
            if np.any(green_region):
                green_enhancement = max(1.15, 1.15 + yellowing * 0.1)
                aged_spectrum[green_region] *= green_enhancement
            aged_spectrum *= 0.85

        aged_spectrum = np.clip(aged_spectrum, 0.0, 1.0)
        original_rgb = self._spectrum_to_rgb(mixed_spectrum)
        aged_rgb = self._spectrum_to_rgb(aged_spectrum)
        
        return original_rgb, aged_rgb
    
    def generate_spectral_aging_data(self, pigment_indices, mixing_ratios,
                       years=100, spectral_bands=31, environmental_conditions=None):
        """
        Generate spectral data with aging effects for use with OpticalProperties module.
        
        Args:
            pigment_indices: Array of pigment indices
            mixing_ratios: Array of mixing ratios
            years: Simulation time in years
            spectral_bands: Number of spectral bands
            environmental_conditions: Optional dict with environmental factors
            
        Returns:
            Dictionary with aging spectral data for visualization
        """
        aging_simulation = self.simulate_aging_effects(
            pigment_indices, mixing_ratios, years, environmental_conditions
        )

        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().numpy()
        else:
            indices = np.array(pigment_indices)

        if mixing_ratios is None:
            ratios = np.ones(len(indices)) / len(indices)
        else:
            if isinstance(mixing_ratios, torch.Tensor):
                ratios = mixing_ratios.detach().cpu().numpy()
            else:
                ratios = np.array(mixing_ratios)

            if np.sum(ratios) > 0:
                ratios = ratios / np.sum(ratios)
            else:
                ratios = np.ones(len(indices)) / len(indices)

        wavelengths = np.linspace(400, 700, spectral_bands)
        aged_spectral = np.zeros(spectral_bands)
        original_spectral = np.zeros(spectral_bands)

        darkening = aging_simulation['aging_effects']['darkening']
        yellowing = aging_simulation['aging_effects']['yellowing']
        fading = aging_simulation['aging_effects']['fading']

        for i, pigment_id in enumerate(indices):
            current_ratio = ratios[i]
            if isinstance(current_ratio, np.ndarray):
                current_ratio = np.mean(current_ratio)
                    
            if current_ratio < 0.05:
                continue
            
            pigment_id = self._extract_scalar(pigment_id)
            if pigment_id in self.spectral_data:
                base_spectral = np.array(self.spectral_data[pigment_id]['reflectance'])
                if len(base_spectral) != spectral_bands:
                    old_wavelengths = np.linspace(400, 700, len(base_spectral))
                    base_spectral = np.interp(wavelengths, old_wavelengths, base_spectral)

                original_spectral += base_spectral * ratios[i]
                pigment_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")

                if pigment_name in aging_simulation['pigment_specific_effects']:
                    specific_effect = aging_simulation['pigment_specific_effects'][pigment_name]
                    
                    aged_pigment_spectral = base_spectral.copy()
                    yellowing_factor = specific_effect['yellowing']

                    if yellowing_factor > 0:
                        blue_region = wavelengths < 480
                        yellow_region = (wavelengths >= 550) & (wavelengths < 600)
                        
                        yellowing_mask = np.ones_like(aged_pigment_spectral)
                        yellowing_mask[blue_region] -= yellowing_factor * 1.5
                        yellowing_mask[yellow_region] += yellowing_factor * 0.7
                        
                        aged_pigment_spectral *= yellowing_mask
                    
                    darkening_factor = specific_effect['darkening']
                    if darkening_factor > 0:
                        darkening_mask = 1.0 - darkening_factor * np.linspace(0.8, 1.0, spectral_bands)
                        aged_pigment_spectral *= darkening_mask

                    fading_factor = specific_effect['fading']
                    if fading_factor > 0:
                        for j in range(len(aged_pigment_spectral)):
                            aged_pigment_spectral[j] = (
                                aged_pigment_spectral[j] * (1.0 - fading_factor * 0.7) + 
                                fading_factor * 0.7
                            )

                    aged_spectral += aged_pigment_spectral * ratios[i]
                else:
                    aged_pigment_spectral = base_spectral.copy()
                    blue_region = wavelengths < 480
                    yellow_region = (wavelengths >= 550) & (wavelengths < 600)
                    
                    yellowing_mask = np.ones_like(aged_pigment_spectral)
                    yellowing_mask[blue_region] -= yellowing * 1.5
                    yellowing_mask[yellow_region] += yellowing * 0.7
                    
                    aged_pigment_spectral *= yellowing_mask
                    darkening_mask = 1.0 - darkening * np.linspace(0.8, 1.0, spectral_bands)
                    aged_pigment_spectral *= darkening_mask

                    for j in range(len(aged_pigment_spectral)):
                        aged_pigment_spectral[j] = (
                            aged_pigment_spectral[j] * (1.0 - fading * 0.7) + 
                            fading * 0.7
                        )

                    aged_spectral += aged_pigment_spectral * ratios[i]

        original_spectral = np.clip(original_spectral, 0.0, 1.0)
        aged_spectral = np.clip(aged_spectral, 0.0, 1.0)

        original_rgb = self._spectrum_to_rgb(original_spectral)
        aged_rgb = self._spectrum_to_rgb(aged_spectral)

        original_lab = self._rgb_to_lab(original_rgb)
        aged_lab = self._rgb_to_lab(aged_rgb)
        delta_e = self._calculate_ciede2000(original_lab, aged_lab)

        return {
            'wavelengths': wavelengths.tolist(),
            'original_spectral': original_spectral.tolist(),
            'aged_spectral': aged_spectral.tolist(),
            'original_rgb': original_rgb,
            'aged_rgb': aged_rgb,
            'delta_e': delta_e,
            'aging_simulation': aging_simulation
        }
    
    def simulate_aging(self, pigment_list, age_years=100, conditions=None):
        """
        Simulate aging for a list of pigment configurations.
        Adapter method used by OpticalProperties to get aging effects.
        
        Args:
            pigment_list: List of dictionaries with 'indices' and 'ratios' keys
            age_years: Simulation time in years
            conditions: Optional environmental conditions
            
        Returns:
            List of aged spectral data
        """
        if conditions is None:
            conditions = {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3
            }

        device = None
        if pigment_list and isinstance(pigment_list[0].get('indices'), torch.Tensor):
            device = pigment_list[0]['indices'].device
        if device is None:
            device = torch.device('cpu')

        aged_spectral_list = []
        for config in pigment_list:
            pigment_indices = config['indices']
            mixing_ratios = config.get('ratios', None)

            aging_data = self.generate_spectral_aging_data(
                pigment_indices,
                mixing_ratios,
                age_years,
                31, 
                conditions
            )
            
            if aging_data and 'aged_spectral' in aging_data:
                aged_spectral_list.append(torch.tensor(aging_data['aged_spectral'], device=device))
            else:
                if hasattr(pigment_indices, 'shape') and len(pigment_indices.shape) > 0:
                    pid = self._extract_scalar(pigment_indices[0])
                else:
                    pid = self._extract_scalar(pigment_indices)
                try:
                    wavelengths, reflectance = self.pigment_db.get_spectral_reflectance(pid)
                    aged_spectral_list.append(torch.tensor(reflectance, device=device))
                except:
                    aged_spectral_list.append(torch.ones(31, device=device) * 0.5)
        
        return aged_spectral_list
