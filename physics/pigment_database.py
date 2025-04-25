import torch
from collections import defaultdict

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