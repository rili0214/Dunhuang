import torch
import numpy as np
from pigment_database import DunhuangPigmentDB

# ===== Thermodynamic Validator =====
class ThermodynamicValidator:
    """
    Validator for pigment mixture stability based on chemical thermodynamics,
    reaction kinetics, spectral validation, non-linear aging models, quantum 
    efficiency, and nanoscale physics for Dunhuang mural pigments.
    """
    def __init__(self, pigment_db):
        pigment_db = DunhuangPigmentDB()
        self.pigment_db = pigment_db
        self._create_pigment_mappings()
        self.reaction_parameters = self._init_reaction_parameters()
        self.quantum_efficiency = self._init_quantum_efficiency()
        self.nanoscale_properties = self._init_nanoscale_properties()
        self.microbial_susceptibility = self._init_microbial_susceptibility()
        self.ph_sensitivity = self._init_ph_sensitivity()
        self.historical_data = self._init_historical_data()
        self.binder_compatibility = self._init_binder_compatibility()
        self.spectral_data = self._init_spectral_data()
    
    def _create_pigment_mappings(self):
        """Create optimized pigment mappings for faster lookups"""
        self.id_to_chemical = {}
        self.name_to_id = {}
        
        for i, pigment in enumerate(self.pigment_db.pigments):
            if i < len(self.pigment_db.pigments):
                formula = pigment.get("formula", "")
                self.id_to_chemical[i] = formula
                self.name_to_id[pigment["name"]] = i

    def _init_reaction_parameters(self):
        """
        Initialize Arrhenius parameters for chemical reactions
        
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
        Initialize nanoscale physical properties with improved accuracy
        
        Format: pigment_id: {particle_size, surface_energy, zeta_potential, aggregation_tendency}
        """
        properties = {}

        for i in range(min(len(self.pigment_db.pigments), 35)):
            try:
                particle_data = self.pigment_db.get_particle_data(i)
                size = particle_data["size"]["mean"]
                properties[i] = {
                    "particle_size": size,
                    "surface_energy": 40 + 30 * (1/size),  # Smaller particles have higher surface energy
                    "zeta_potential": -20 - 10 * (1/size),  # Smaller particles have more negative potential
                    "aggregation_tendency": 0.3 + 0.3 * (1/size)  # Smaller particles aggregate more
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
                    [0, 1, 2, 3, 4, 5],    # Expanded white pigment group
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
        
        This implements the full CIEDE2000 formula with all correction terms
        
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
        
        # Apply gamma correction if RGB is already linear
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
        # D65 reference white
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
        
        # Convert IDs to integers if they are tensors
        if hasattr(pigment1, 'item'):
            pigment1 = pigment1.item()
        if hasattr(pigment2, 'item'):
            pigment2 = pigment2.item()
        
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
            (0.4 * size_ratio) +                 # Size mismatch causes poor packing
            (0.3 * surface_energy_diff / 50) +   # Surface energy mismatch affects wetting
            (0.1 * (1 - zeta_diff / 40)) +       # Similar zeta potential can lead to aggregation
            (0.2 * aggregation_potential)        # Direct measure of aggregation tendency
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
        pigment_indices = [p.item() if hasattr(p, 'item') else p for p in pigment_indices]
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
        
        # Calculate risk score with humidity factor
        humidity_factor = 0.5 * np.exp(humidity - 0.6)
        
        # Base risk from organic content and nutrient value
        base_risk = ((0.5 * avg_organic) + (0.5 * avg_nutrients)) * humidity_factor
        
        # Biocidal properties reduce risk non-linearly
        biocide_protection = 1.0 - np.sqrt(avg_biocide)
        
        # Calculate mitigated risk
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
        if hasattr(pigment_id, 'item'):
            pigment_id = pigment_id.item()
        if pigment_id not in self.quantum_efficiency:
            qe_values = {"uv_a": 0.001, "uv_b": 0.005, "visible": 0.0002}
            mechanism = "unknown"
        else:
            qe_values = self.quantum_efficiency[pigment_id]
            mechanisms = {
                "uv_b": "UV-B photolysis",
                "uv_a": "UV-A induced oxidation",
                "visible": "visible light photochemical degradation"
            }
            primary = max(qe_values, key=qe_values.get)
            mechanism = mechanisms[primary]

        # Low light: mostly visible (0.2 UV-A, 0.1 UV-B, 0.7 visible)
        # Medium light: mixed (0.3 UV-A, 0.2 UV-B, 0.5 visible)
        # High light: more UV (0.4 UV-A, 0.3 UV-B, 0.3 visible)
        if light_exposure < 0.4:  # Low light
            uv_a_weight, uv_b_weight, visible_weight = 0.2, 0.1, 0.7
        elif light_exposure < 0.7:  # Medium light
            uv_a_weight, uv_b_weight, visible_weight = 0.3, 0.2, 0.5
        else:  # High light
            uv_a_weight, uv_b_weight, visible_weight = 0.4, 0.3, 0.3
            
        # Calculate weighted degradation factor
        degradation_factor = (
            uv_a_weight * qe_values["uv_a"] +
            uv_b_weight * qe_values["uv_b"] +
            visible_weight * qe_values["visible"]
        ) * light_exposure * 50
        
        # Apply non-linear saturation effect for high light levels
        if light_exposure > 0.7:
            degradation_factor *= (1 + (light_exposure - 0.7) * 0.5)
        
        # Apply dosage effect
        degradation_factor = 1.0 - np.exp(-degradation_factor)
        
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

                if ratios[i] < 0.03 or ratios[j] < 0.03:
                    continue

                rate, reaction_type = self._calculate_reaction_rate(pigment1, pigment2, temperature)
                
                if rate > 0:
                    # Calculate instability based on reaction rate, concentrations and stoichiometry
                    concentration_factor = min(ratios[i], ratios[j])
                    if reaction_type == "chemical_reaction":
                        severity = min(1.0, rate / 1e3)  # Threshold of 1e3 for full severity
                    elif reaction_type == "catalytic_degradation":
                        severity = min(1.0, rate / 5e2)  # Lower threshold for degradation
                    else:
                        severity = min(1.0, rate / 2e3)  # Default threshold
                    
                    # Calculate instability contribution
                    instability = concentration_factor * severity * 0.5
                    base_stability_score -= instability
                    
                    # Generate warning message
                    p1_name = self.pigment_db.id_to_name.get(pigment1, f"Pigment {pigment1}")
                    p2_name = self.pigment_db.id_to_name.get(pigment2, f"Pigment {pigment2}")
                    
                    # Get reaction description
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
                    
                    # Apply stability penalty
                    nano_instability = impact * 0.3 * min(ratios[i], ratios[j])
                    base_stability_score -= nano_instability
                    
                    # Determine dominant nanoscale issue
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
                if ratios[i] < 0.05:  
                    continue
                    
                degradation_factor, mechanism_info = self._calculate_quantum_efficiency_factor(
                    pigment_id, light_exposure
                )
                
                if degradation_factor > 0.2:  
                    photo_impact = degradation_factor * ratios[i] * 0.4
                    base_stability_score -= photo_impact
                    
                    p_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
                    warning = f"Photodegradation risk: {p_name} - {mechanism_info['mechanism']}"
                    warnings.append(warning)
                    
                    detailed_issues.append({
                        "type": "photodegradation",
                        "pigment": p_name,
                        "mechanism": mechanism_info["mechanism"],
                        "degradation_factor": degradation_factor,
                        "impact": photo_impact
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
                if ratios[i] < 0.05: 
                    continue
                
                if pigment_id in binder_data:
                    compatibility = binder_data[pigment_id]
                    if compatibility < 0.7:
                        penalty = (0.7 - compatibility) * ratios[i] * 0.5
                        binder_penalty += penalty
                        
                        p_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
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
            if pigment_id in self.ph_sensitivity:
                sens_data = self.ph_sensitivity[pigment_id]
                if sens_data["acid"] > 0.5:
                    total_acidity += sens_data["acid"] * ratios[i]
                if sens_data["alkaline"] > 0.5:
                    total_alkalinity += sens_data["alkaline"] * ratios[i]
        
        # Penalize mixtures with both acidic and alkaline components
        if total_acidity > 0.2 and total_alkalinity > 0.2:
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
            # Check if this mixture matches any historical compatible groups
            for group in data["compatible_groups"]:
                # Calculate overlap with this historical group
                match_count = sum(1 for p in indices if p in group)
                if match_count > 1 and match_count == len(indices):
                    historical_bonus = 0.1  # Bonus for historically proven combination
                    detailed_issues.append({
                        "type": "historical_match",
                        "period": period,
                        "impact": historical_bonus
                    })
            
            # Check if mixture contains historically avoided combinations
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
        
        # Apply non-linear synergistic effects for multiple issues
        if len(detailed_issues) > 1:
            impacts = [issue.get('impact', 0) for issue in detailed_issues]
            impacts.sort(reverse=True)
            
            # Apply synergistic penalty
            # Base formula: synergistic_penalty = sum(impacts^1.2) - sum(impacts)
            synergy_penalty = sum(abs(impact)**1.2 for impact in impacts) - sum(impacts)
            issue_count_factor = 1.0 + (len(detailed_issues) - 1) * 0.1
            synergy_penalty *= issue_count_factor
            synergy_penalty = min(0.3, max(0, synergy_penalty)) 
            
            # Apply synergy penalty
            base_stability_score -= synergy_penalty
            
            # Add explanation
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
        """Identify the reaction product between two pigments"""
        # This is a simplified lookup, I will add more in the future, if have time......
        # Lead pigments with sulfur compounds
        if (pigment1 == 3 and pigment2 in [10, 11, 13]) or (pigment2 == 3 and pigment1 in [10, 11, 13]):
            return "black lead sulfide"
        # Copper compounds with sulfur compounds
        elif (pigment1 in [17, 21, 22] and pigment2 in [10, 11, 13, 23, 24]) or \
             (pigment2 in [17, 21, 22] and pigment1 in [10, 11, 13, 23, 24]):
            return "copper sulfide (black)"
        else:
            return "degradation products"
    
    def _validate_spectral_stability(self, pigment_indices, mixing_ratios):
        """Validate spectral stability of mixture"""
        mixed_spectrum = np.zeros(31)

        if isinstance(pigment_indices, torch.Tensor):
            pigment_indices = pigment_indices.detach().cpu().numpy()
        if isinstance(mixing_ratios, torch.Tensor):
            mixing_ratios = mixing_ratios.detach().cpu().numpy()
        
        # Weight each pigment's spectrum by its ratio
        for i, pigment_id in enumerate(pigment_indices):
            if pigment_id in self.spectral_data:
                reflectance = np.array(self.spectral_data[pigment_id]['reflectance'])
                weighted_spectrum = reflectance * mixing_ratios[i]
                mixed_spectrum += weighted_spectrum

        if np.max(mixed_spectrum) > 1.0:
            mixed_spectrum = mixed_spectrum / np.max(mixed_spectrum)
        
        # Simulate aged spectrum with more pronounced effects
        aged_spectrum = mixed_spectrum.copy()
        
        # Check for known problematic pigment combinations
        has_lead = any(p in [3, 5, 9, 14] for p in pigment_indices)
        has_sulfur = any(p in [10, 11, 13, 23, 24] for p in pigment_indices)
        has_copper = any(p in [17, 21, 22] for p in pigment_indices)
        
        # Apply stronger aging factors for problematic combinations
        if has_lead and has_sulfur:
            darkening_factor = 0.5  # Significant darkening due to lead sulfide formation
            yellowing_factor = 0.2
        elif has_copper and has_sulfur:
            darkening_factor = 0.4  # Significant darkening due to copper sulfide formation
            yellowing_factor = 0.15
        else:
            darkening_factor = 0.2  # Default aging factors
            yellowing_factor = 0.15
        
        # Apply general darkening across spectrum
        wavelengths = np.linspace(400, 700, 31)
        aged_spectrum *= (1.0 - darkening_factor)
        
        # Apply wavelength-specific changes for yellowing
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
        
        # More nuanced interpretation of the difference
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
            
        # Check interlayer interactions
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
        
        # Calculate interface penalty
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
        
        # Calculate overall score
        overall_score = base_score - interface_penalty
        overall_score = max(0.0, min(1.0, overall_score))
        
        return {
            "layer_stability": layer_stability,
            "interlayer_issues": interlayer_issues,
            "overall_stability_score": overall_score,
            "is_stable": overall_score >= 0.65,
            "interface_penalty": interface_penalty
        }
    
    def _check_interlayer_interactions(self, bottom_layer, top_layer, bottom_binder, top_binder):
        """Check for issues between adjacent layers."""
        issues = []
        
        # Calculate temperature
        temperature = 298

        for p1 in bottom_layer:
            for p2 in top_layer:
                # Check for reactions using Arrhenius kinetics
                rate, reaction_type = self._calculate_reaction_rate(p1, p2, temperature)
                
                if rate > 0:
                    # Scale by migration potential between layers
                    migration_factor = self._calculate_migration_potential(p1, p2, bottom_binder, top_binder)
                    if migration_factor > 0.3:
                        p1_name = self.pigment_db.id_to_name.get(p1, f"Pigment {p1}")
                        p2_name = self.pigment_db.id_to_name.get(p2, f"Pigment {p2}")
                        
                        # Calculate effective severity based on reaction rate and migration
                        severity = min(0.9, rate / 1e3) * migration_factor
                        
                        issues.append({
                            "type": "layer_interaction",
                            "pigments": [p1_name, p2_name],
                            "mechanism": reaction_type,
                            "severity": severity,
                            "description": f"Migration and reaction: {self._get_reaction_product(p1, p2)}"
                        })
        
        # Check physical interface compatibility
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
                
            # Check for differential drying/curing rates
            if (bottom_binder in ["drying_oil", "egg_tempera"] and 
                top_binder in ["animal_glue", "gum_arabic"]):
                issues.append({
                    "type": "binder_interaction",
                    "binders": [bottom_binder, top_binder],
                    "severity": 0.5,
                    "description": "Fast-drying layer over slow-drying layer can cause cracking"
                })
        
        # Check for pigment concentration gradient issues
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
        """Calculate the potential for material migration between layers"""
        
        # 1. Estimate solubility factors based on pigment chemistry
        solubility_factor = 0.5 
        
        # Water-soluble pigments
        water_soluble = [30, 31, 32, 33] 
        
        # Oil-soluble pigments
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
            "animal_glue": 0.8,  # High porosity
            "gum_arabic": 0.6,   # Medium porosity
            "egg_tempera": 0.4,  # Low-medium porosity
            "drying_oil": 0.3    # Low porosity
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
                size_factor = 0.9  # Very small particles migrate easily
            elif smallest_size < 3.0:
                size_factor = 0.7  # Small particles
            elif smallest_size < 8.0:
                size_factor = 0.5  # Medium particles
            else:
                size_factor = 0.3  # Large particles rarely migrate
        
        migration_potential = (
            0.4 * solubility_factor +
            0.4 * porosity_factor +
            0.2 * size_factor
        )
        
        return migration_potential
    
    def _is_porous_pigment(self, pigment_id):
        """Check if a pigment tends to create porous paint layers"""
        porous_pigments = [0, 1, 2, 18, 25]  # Kaolin, calcium carbonate, gypsum, lapis, carbon black
        return pigment_id in porous_pigments
    
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
                    problematic_pigments[pigment] += issue["impact"] * 0.5  # Lower weight
        
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
            # Suggest better binder options
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
        
        # If ratios were modified, renormalize
        if ratios_modified:
            ratios = ratios / np.sum(ratios)
            
            # Check stability after corrections
            corrected_stability = self.check_mixture_stability(
                indices, ratios, binder_type, environmental_conditions
            )
            
            # Add evaluation of corrections
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
        """Calculate spectral similarity between two pigments"""
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
        """Suggest alternative binders with better compatibility"""
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
        
        # Sort by compatibility score
        binder_scores.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        return binder_scores
    
    def _is_historically_appropriate(self, pigment_id):
        """Check if pigment was used in historical Dunhuang pigment practice"""
        for period, data in self.historical_data.items():
            for group in data["compatible_groups"]:
                if pigment_id in group:
                    return True
        return False
    
    def _is_historically_appropriate_binder(self, binder_type):
        """Check if binder was used in historical practice"""
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
        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().numpy()
        else:
            indices = np.array(pigment_indices)
        if isinstance(mixing_ratios, torch.Tensor):
            ratios = mixing_ratios.detach().cpu().numpy()
        else:
            ratios = np.array(mixing_ratios)

        if np.sum(ratios) > 0:
            ratios = ratios / np.sum(ratios)

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
        
        # Initialize degradation effects with base values influenced by stability
        base_instability = max(0, 1.0 - stability_score)
        darkening_total = base_instability * 0.3    # Initialize with some darkening for unstable mixtures
        yellowing_total = base_instability * 0.2    # Initialize with some yellowing for unstable mixtures
        fading_total = base_instability * 0.1       # Initialize with some fading for unstable mixtures
        cracking_total = base_instability * 0.2     # Initialize with some cracking for unstable mixtures

        aging_stages = [
            {"fraction": 0.2, "name": "Initial-settling", "days": 365},         # First year
            {"fraction": 0.4, "name": "Medium-term", "days": 365 * 9},          # Years 2-10
            {"fraction": 0.4, "name": "Long-term", "days": 365 * (years - 10)}  # Remaining years
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
            if ratios[i] < 0.05: 
                continue

            try:
                pigment_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
            except (AttributeError, IndexError):
                pigment_name = f"Pigment {pigment_id}"
                
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
        
        # For each stage, calculate degradation
        for stage in aging_stages:
            stage_darkening = 0.0
            stage_yellowing = 0.0
            stage_fading = 0.0
            stage_cracking = 0.0
            time_seconds = stage["days"] * 24 * 3600
            
            # For each pigment, calculate degradation using various mechanisms
            for i, pigment_id in enumerate(indices):
                if ratios[i] < 0.05:
                    continue

                try:
                    pigment_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
                except (AttributeError, IndexError):
                    pigment_name = f"Pigment {pigment_id}"
                
                # Calculate chemical interactions with other pigments
                chemical_interactions = []
                
                for j, other_id in enumerate(indices):
                    if i == j or ratios[j] < 0.05:
                        continue
                    
                    # Calculate reaction rate for each pair
                    rate, reaction_type = self._calculate_reaction_rate(pigment_id, other_id, temperature)
                    
                    if rate > 0:
                        effective_rate = rate * min(ratios[i], ratios[j]) * (1.0 + years / 100.0)
                        chemical_interactions.append({
                            "partner": other_id,
                            "rate": effective_rate,
                            "type": reaction_type
                        })
                
                # Photodegradation effects
                if "light_exposure" in environmental_conditions:
                    light_factor, light_mechanism = self._calculate_quantum_efficiency_factor(
                        pigment_id, environmental_conditions["light_exposure"]
                    )
                    
                    # Add to pigment-specific effects
                    if light_factor > 0.1:
                        if light_mechanism["mechanism"] not in pigment_specific_effects[pigment_name]["mechanisms"]:
                            pigment_specific_effects[pigment_name]["mechanisms"].append(
                                light_mechanism["mechanism"]
                            )
                        
                        if pigment_id in [13, 23, 30, 31, 32]:
                            # These tend to fade with light
                            pigment_specific_effects[pigment_name]["fading"] += light_factor * stage["fraction"]
                            stage_fading += light_factor * ratios[i] * stage["fraction"]
                        elif pigment_id in [10, 11]: 
                            # These tend to darken with light
                            pigment_specific_effects[pigment_name]["darkening"] += light_factor * stage["fraction"]
                            stage_darkening += light_factor * ratios[i] * stage["fraction"]
                
                for interaction in chemical_interactions:
                    other_id = interaction["partner"]
                    
                    try:
                        other_name = self.pigment_db.id_to_name.get(other_id, f"Pigment {other_id}")
                    except (AttributeError, IndexError):
                        other_name = f"Pigment {other_id}"
                    
                    effect_magnitude = 1.0 - np.exp(-interaction["rate"] * time_seconds * 1e-6 * (1 + base_instability))
                    effect_magnitude = min(0.9, effect_magnitude)  # Cap at 90% degradation
                    
                    is_lead_sulfide = ((pigment_id in [3, 5, 9, 14] and other_id in [10, 11, 13, 23, 24]) or
                                    (other_id in [3, 5, 9, 14] and pigment_id in [10, 11, 13, 23, 24]))
                    
                    is_copper_sulfide = ((pigment_id in [17, 21, 22] and other_id in [10, 11, 13, 23, 24]) or
                                        (other_id in [17, 21, 22] and pigment_id in [10, 11, 13, 23, 24]))
                    
                    if is_lead_sulfide:
                        effect_magnitude *= 2.0 
                    
                    if is_copper_sulfide:
                        effect_magnitude *= 1.5
                    
                    if interaction["type"] == "chemical_reaction":
                        pigment_specific_effects[pigment_name]["darkening"] += effect_magnitude * stage["fraction"]
                        stage_darkening += effect_magnitude * ratios[i] * stage["fraction"]

                        mechanism = f"Reaction with {other_name}"
                        if mechanism not in pigment_specific_effects[pigment_name]["mechanisms"]:
                            pigment_specific_effects[pigment_name]["mechanisms"].append(mechanism)
                    
                    elif interaction["type"] == "catalytic_degradation":
                        pigment_specific_effects[pigment_name]["fading"] += effect_magnitude * 0.6 * stage["fraction"]
                        pigment_specific_effects[pigment_name]["yellowing"] += effect_magnitude * 0.4 * stage["fraction"]
                        
                        stage_fading += effect_magnitude * 0.6 * ratios[i] * stage["fraction"]
                        stage_yellowing += effect_magnitude * 0.4 * ratios[i] * stage["fraction"]

                        mechanism = f"Catalyzed by {other_name}"
                        if mechanism not in pigment_specific_effects[pigment_name]["mechanisms"]:
                            pigment_specific_effects[pigment_name]["mechanisms"].append(mechanism)
                            
                    elif interaction["type"] == "acid_base_reaction":  
                        pigment_specific_effects[pigment_name]["yellowing"] += effect_magnitude * 0.3 * stage["fraction"]
                        pigment_specific_effects[pigment_name]["cracking"] += effect_magnitude * 0.5 * stage["fraction"]
                        
                        stage_yellowing += effect_magnitude * 0.3 * ratios[i] * stage["fraction"]
                        stage_cracking += effect_magnitude * 0.5 * ratios[i] * stage["fraction"]
                        
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
            
            # Accumulate stage effects
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
        
        # Simulate spectral changes using modern color science methods
        original_rgb, aged_rgb = self._calculate_aged_color(
            indices, ratios, 
            darkening_total, yellowing_total, fading_total
        )
        
        # Calculate color difference using improved CIEDE2000
        original_lab = self._rgb_to_lab(original_rgb)
        aged_lab = self._rgb_to_lab(aged_rgb)
        delta_e = self._calculate_ciede2000(original_lab, aged_lab)

        if stability_score < 0.5 and delta_e < 10:
            delta_e = 10 + (0.5 - stability_score) * 40 
        
        # Determine overall condition after aging
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

        has_lead = any(p in [3, 5, 9, 14] for p in indices)
        has_sulfur = any(p in [10, 11, 13, 23, 24] for p in indices)
        if has_lead and has_sulfur and darkening_total > 0.2:
            explanation += " Lead compounds will react with sulfur-containing pigments to form black lead sulfide."
        
        # Check for copper-sulfur reaction
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

        pigment_indices = [p.item() if hasattr(p, 'item') else p for p in pigment_indices]

        for i, pigment_id in enumerate(pigment_indices):
            if pigment_id in self.spectral_data:
                reflectance = np.array(self.spectral_data[pigment_id]['reflectance'])
                weighted_spectrum = reflectance * mixing_ratios[i]
                mixed_spectrum += weighted_spectrum

        if np.max(mixed_spectrum) > 1.0:
            mixed_spectrum = mixed_spectrum / np.max(mixed_spectrum)
        
        # Apply aging effects to spectrum
        aged_spectrum = mixed_spectrum.copy()
        
        # Apply wavelength-dependent aging effects for more realism
        wavelengths = np.linspace(400, 700, 31)
        
        # 1. Darkening - reduce reflectance more in some regions
        darkening_factor = darkening * 2.0 
        darkening_mask = 1.0 - (darkening_factor * np.linspace(0.7, 1.3, 31)) 
        darkening_mask = np.clip(darkening_mask, 0.0, 1.0)
        aged_spectrum *= darkening_mask
        
        # 2. Yellowing - reduces blue reflectance, increases yellow
        blue_region = wavelengths < 480
        yellow_region = (wavelengths >= 550) & (wavelengths < 600)
        yellowing_factor = yellowing * 2.5
        if np.any(blue_region):
            aged_spectrum[blue_region] *= (1.0 - yellowing_factor * 1.5)
        if np.any(yellow_region):
            aged_spectrum[yellow_region] *= (1.0 + yellowing_factor * 1.2)
        
        # 3. Fading - pull toward white (1.0) but preserve some color character
        if fading > 0.05:
            fade_weights = 0.3 + 0.7 * np.sin(np.linspace(0, np.pi, 31))
            fade_amount = fading * 2.0 * fade_weights
            for i in range(len(aged_spectrum)):
                aged_spectrum[i] = aged_spectrum[i] * (1.0 - fade_amount[i]) + fade_amount[i]
        
        # 4. Add chemical-specific shifts for common pigment interactions
        lead_white_idx = 3
        has_lead = any(p in [3, 5, 9, 14] for p in pigment_indices)
        sulfur_pigments = [10, 11, 13, 23, 24]
        has_sulfur = any(p in sulfur_pigments for p in pigment_indices)
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
        Generate spectral data with aging effects for use with OpticalProperties module
        
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
        if isinstance(mixing_ratios, torch.Tensor):
            ratios = mixing_ratios.detach().cpu().numpy()
        else:
            ratios = np.array(mixing_ratios)

        if np.sum(ratios) > 0:
            ratios = ratios / np.sum(ratios)

        wavelengths = np.linspace(400, 700, spectral_bands)
        aged_spectral = np.zeros(spectral_bands)
        original_spectral = np.zeros(spectral_bands)

        darkening = aging_simulation['aging_effects']['darkening']
        yellowing = aging_simulation['aging_effects']['yellowing']
        fading = aging_simulation['aging_effects']['fading']
        
        # For each pigment, calculate its spectral contribution
        for i, pigment_id in enumerate(indices):
            if ratios[i] < 0.01:  
                continue

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
