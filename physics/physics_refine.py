"""
Integrated physics refinement module that combines color analysis, 
pigment database, thermodynamic validation, and optical properties 
for physically accurate simulation of historical pigments.

This module provides a complete pipeline for physically accurate simulation 
of historically accurate pigments, including:
1. Analyzing images to identify historically accurate pigments
2. Validating the chemical stability of pigment mixtures
3. Simulating aging and environmental effects
4. Generating physically accurate rendering properties

@Author: Yuming Xie
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union
import warnings
import json
import csv
import io

from color_pig_map import ColorToPigmentMapper
from pigment_database import DunhuangPigmentDB
from themodynamicval import ThermodynamicValidator
from optical_properties import OpticalProperties

class PhysicsRefinementModule:
    """
    Integrated physics refinement module that combines color analysis, 
    pigment database, thermodynamic validation, and optical properties 
    for physically accurate simulation of historical pigments.
    """
    def __init__(self, 
                num_pigments: int = 35,
                spectral_bands: int = 31,
                swin_feat_dim: int = 768,
                color_feat_dim: int = 128,
                device: Optional[torch.device] = None):
        """
        Initialize the Physics Refinement Module with all components.
        
        Args:
            num_pigments: Number of pigments in the database
            spectral_bands: Number of spectral bands for spectral rendering
            swin_feat_dim: Feature dimension for Swin Transformer input features
            color_feat_dim: Feature dimension for color analysis
            device: Computation device (CPU/GPU)
        """
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_pigments = num_pigments
        self.spectral_bands = spectral_bands
        self.pigment_db = DunhuangPigmentDB(spectral_resolution=spectral_bands)
        self.validator = ThermodynamicValidator(pigment_db=self.pigment_db)
        self.color_mapper = ColorToPigmentMapper(
            swin_feat_dim=swin_feat_dim,
            color_feat_dim=color_feat_dim,
            num_pigments=num_pigments,
            spectral_bands=spectral_bands,
            device=self.device
        )
        self.optical_props = OpticalProperties(
            num_pigments=num_pigments,
            pigment_embedding_dim=64,
            spectral_bands=spectral_bands,
            device=self.device
        )
        
        #print(f"Physics Refinement Module initialized on {self.device}")
        
    def analyze_image(self, 
                     image: torch.Tensor, 
                     features: Optional[torch.Tensor] = None, 
                     n_suggestions: int = 5) -> Dict:
        """
        Analyze an image to identify historically accurate pigments.
        
        Args:
            image: Input image tensor [B, C, H, W]
            features: Optional pre-extracted features
            n_suggestions: Number of pigment suggestions to return
            
        Returns:
            Dictionary with pigment analysis, including:
            - pigment_probabilities: Probability distribution over pigments
            - color_clusters: Identified color clusters in the image
            - pigment_suggestions: Detailed pigment suggestions
            - brdf_params: Parameters for physically-based rendering
            - spectral_data: Full spectral analysis
        """
        image = image.to(self.device)
        if features is not None:
            features = features.to(self.device)

        analysis = self.color_mapper.analyze_image_pigments(image, features)

        if n_suggestions > 0 and n_suggestions != 5: 
            suggestions = self.color_mapper.get_pigment_suggestions(image, n_suggestions)
            analysis['pigment_suggestions'] = suggestions['suggestions']
        
        return analysis
    
    def validate_pigment_mixture(self, 
                            pigment_indices: torch.Tensor,
                            mixing_ratios: Optional[torch.Tensor] = None,
                            binder_type: Optional[str] = None,
                            environmental_conditions: Optional[Dict] = None,
                            temperature: float = 298) -> Dict:
        """
        Validate the chemical and physical stability of a pigment mixture.
        """
        pigment_indices = pigment_indices.to(self.device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(self.device)
        if len(pigment_indices.shape) > 1 and pigment_indices.shape[0] == 1:
            pigment_indices = pigment_indices.squeeze(0)
        if mixing_ratios is not None and len(mixing_ratios.shape) > 1 and mixing_ratios.shape[0] == 1:
            mixing_ratios = mixing_ratios.squeeze(0)
        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3
            }

        stability_info = self.validator.check_mixture_stability(
            pigment_indices, 
            mixing_ratios,
            binder_type, 
            environmental_conditions,
            temperature
        )
        
        return stability_info
    
    def suggest_mixture_improvements(self,
                                  pigment_indices: torch.Tensor,
                                  mixing_ratios: Optional[torch.Tensor] = None,
                                  binder_type: Optional[str] = None,
                                  environmental_conditions: Optional[Dict] = None) -> Dict:
        """
        Suggest improvements to enhance the stability of a pigment mixture.
        
        Args:
            pigment_indices: Indices of pigments [batch_size, n_pigments]
            mixing_ratios: Optional mixing ratios [batch_size, n_pigments]
            binder_type: Optional binder material
            environmental_conditions: Optional dict with environmental factors
            
        Returns:
            Dictionary with suggested improvements, including:
            - corrected_ratios: Improved mixing ratios
            - original_stability: Original stability assessment
            - corrected_stability: Projected stability after changes
            - suggestions: List of specific improvement suggestions
        """
        pigment_indices = pigment_indices.to(self.device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(self.device)

        suggestions = self.validator.suggest_corrections(
            pigment_indices,
            mixing_ratios,
            binder_type,
            environmental_conditions
        )
        
        return suggestions
    
    def simulate_aging(self,
                    pigment_indices: torch.Tensor,
                    mixing_ratios: Optional[torch.Tensor] = None,
                    years: int = 100,
                    environmental_conditions: Optional[Dict] = None,
                    temperature: float = 298) -> Dict:
        """
        Simulate the aging effects on pigments over time.
        """
        pigment_indices = pigment_indices.to(self.device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(self.device)
        if len(pigment_indices.shape) > 1 and pigment_indices.shape[0] == 1:
            pigment_indices = pigment_indices.squeeze(0)
        if mixing_ratios is not None and len(mixing_ratios.shape) > 1 and mixing_ratios.shape[0] == 1:
            mixing_ratios = mixing_ratios.squeeze(0)
        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3
            }

        aging_simulation = self.validator.simulate_aging_effects(
            pigment_indices,
            mixing_ratios,
            years,
            environmental_conditions,
            temperature
        )

        spectral_aging = self.validator.generate_spectral_aging_data(
            pigment_indices,
            mixing_ratios,
            years,
            self.spectral_bands,
            environmental_conditions
        )

        aging_simulation['spectral_aging'] = spectral_aging
        
        return aging_simulation
    
    def calculate_optical_properties(self,
                               pigment_indices: torch.Tensor,
                               mixing_ratios: Optional[torch.Tensor] = None,
                               age_years: int = 0,
                               particle_data: Optional[Dict] = None,
                               fluorescence_params: Optional[Dict] = None,
                               microstructure_params: Optional[Dict] = None,
                               layered_configuration: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate physically-based optical properties for rendering.
        
        Args:
            pigment_indices: Indices of pigments [batch_size, n_pigments]
            mixing_ratios: Optional mixing ratios [batch_size, n_pigments]
            age_years: Optional aging simulation in years
            particle_data: Optional particle size data
            fluorescence_params: Optional fluorescence parameters
            microstructure_params: Optional microstructure parameters
            layered_configuration: Optional configuration for layered rendering
            
        Returns:
            Dictionary with comprehensive optical properties, including:
            - rgb: RGB color values
            - spectral: Full spectral reflectance data
            - brdf_params: Parameters for physically-based rendering
            - optical_params: Additional optical parameters
        """
        thermodynamic_validator = self.validator
        pigment_indices = pigment_indices.to(self.device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(self.device)
        if particle_data is None and not layered_configuration:
            mean_sizes = []
            for i in range(pigment_indices.shape[0]):
                if len(pigment_indices.shape) > 1:
                    sample_sizes = []
                    for j in range(pigment_indices.shape[1]):
                        pid = pigment_indices[i, j].item()
                        try:
                            particle_data = self.pigment_db.get_particle_data(pid)
                            sample_sizes.append(particle_data["size"]["mean"])
                        except (KeyError, ValueError, IndexError):
                            sample_sizes.append(5.0) 
                    if mixing_ratios is not None:
                        weights = mixing_ratios[i].cpu().detach().numpy()
                        mean_sizes.append(np.average(sample_sizes, weights=weights))
                    else:
                        mean_sizes.append(np.mean(sample_sizes))
                else:
                    pid = pigment_indices[i].item()
                    try:
                        particle_data = self.pigment_db.get_particle_data(pid)
                        mean_sizes.append(particle_data["size"]["mean"])
                    except (KeyError, ValueError, IndexError):
                        mean_sizes.append(5.0)
                        
            particle_data = {
                "mean": torch.tensor(mean_sizes, device=self.device),
                "std": torch.tensor([1.0] * len(mean_sizes), device=self.device)
            }

        precomputed_aging = None
        aging_data = None
        if age_years > 0:
            if thermodynamic_validator is not None:
                aging_data = thermodynamic_validator.generate_spectral_aging_data(
                    pigment_indices,
                    mixing_ratios,
                    age_years,
                    self.spectral_bands
                )
                if aging_data and 'aged_spectral' in aging_data:
                    precomputed_aging = aging_data['aged_spectral']

        optical_result = self.optical_props(
            pigment_indices,
            mixing_ratios,
            precomputed_aging,
            particle_data,
            fluorescence_params,
            microstructure_params,
            layered_configuration
        )

        if precomputed_aging is not None:
            aged_rgb, aged_spectral = self.optical_props.apply_spectral_aging(
                optical_result['spectral'], precomputed_aging
            )
            optical_result['spectral'] = aged_spectral
            optical_result['rgb'] = aged_rgb 
            optical_result['aged'] = True
            optical_result['age_years'] = age_years

        if age_years > 0 and precomputed_aging is not None:
            original_spectral = optical_result.get('spectral', None)
            if original_spectral is not None and aging_data is not None:
                if 'aged_spectral' in aging_data:
                    consistency_metrics = self._calculate_spectral_consistency(
                        original_spectral,
                        aging_data['aged_spectral'],
                        optical_result['spectral']
                    )

                    optical_result['aging_consistency'] = consistency_metrics

                    if not consistency_metrics['within_tolerance']:
                        warnings.warn(
                            f"Aging simulations diverge significantly: " 
                            f"ΔE={consistency_metrics['delta_e_between_methods']:.2f}"
                        )
            
        return optical_result
    
    def analyze_aging_uncertainty(self, pigment_indices, mixing_ratios=None, years=100, n_samples=100):
        """
        Analyze aging uncertainty using Monte Carlo simulation.
        
        Args:
            pigment_indices: Pigment indices
            mixing_ratios: Optional mixing ratios
            years: Simulation time in years
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with uncertainty analysis and visualization data
        """
        device = self.device
        pigment_indices = pigment_indices.to(device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(device)

        monte_carlo_results = self.validator.monte_carlo_aging_simulation(
            pigment_indices,
            mixing_ratios,
            years,
            n_samples
        )

        spectral_bands = self.validator.generate_spectral_uncertainty_bands(
            pigment_indices,
            mixing_ratios,
            years,
            min(50, n_samples),
            self.spectral_bands
        )

        if spectral_bands:
            mean_aged_spectral = torch.tensor(
                spectral_bands['mean_aged_spectral'], 
                device=device
            )
            
            optical_result = self.calculate_optical_properties(
                pigment_indices,
                mixing_ratios,
                age_years=years,
                precomputed_aging=mean_aged_spectral
            )
            
            return {
                'monte_carlo_results': monte_carlo_results,
                'spectral_uncertainty': spectral_bands,
                'optical_properties': optical_result,
                'sensitivity_analysis': monte_carlo_results.get('sensitivity', {}),
                'confidence_interval': monte_carlo_results.get('confidence_interval_95', {})
            }
        else:
            return {
                'monte_carlo_results': monte_carlo_results,
                'error': 'Failed to generate spectral uncertainty bands'
            }
    
    def _calculate_spectral_consistency(self, original_spectral, thermo_aged_spectral, optical_aged_spectral):
        """
        Calculate consistency between different aging simulation methods.
        
        Args:
            original_spectral: Original spectral values [batch_size, spectral_bands]
            thermo_aged_spectral: Aged spectral from thermodynamic validator
            optical_aged_spectral: Aged spectral from optical properties
            
        Returns:
            Dictionary with consistency metrics
        """
        device = original_spectral.device
  
        if isinstance(thermo_aged_spectral, np.ndarray):
            thermo_aged_spectral = torch.tensor(thermo_aged_spectral, dtype=torch.float32, device=device)
        elif isinstance(thermo_aged_spectral, list):
            thermo_aged_spectral = torch.tensor(thermo_aged_spectral, dtype=torch.float32, device=device)
        if len(thermo_aged_spectral.shape) == 1:
            thermo_aged_spectral = thermo_aged_spectral.unsqueeze(0)
        if len(optical_aged_spectral.shape) == 1:
            optical_aged_spectral = optical_aged_spectral.unsqueeze(0)

        original_rgb = self.optical_props._spectral_to_rgb_conversion(original_spectral)
        thermo_rgb = self.optical_props._spectral_to_rgb_conversion(thermo_aged_spectral)
        optical_rgb = self.optical_props._spectral_to_rgb_conversion(optical_aged_spectral)

        original_lab = self.validator._rgb_to_lab(original_rgb[0].cpu().detach().numpy())
        thermo_lab = self.validator._rgb_to_lab(thermo_rgb[0].cpu().detach().numpy())
        optical_lab = self.validator._rgb_to_lab(optical_rgb[0].cpu().detach().numpy())

        delta_e_thermo = self.validator._calculate_ciede2000(original_lab, thermo_lab)
        delta_e_optical = self.validator._calculate_ciede2000(original_lab, optical_lab)
        delta_e_between = self.validator._calculate_ciede2000(thermo_lab, optical_lab)

        spectral_rmse = torch.sqrt(torch.mean((thermo_aged_spectral - optical_aged_spectral) ** 2))
        max_diff = torch.max(torch.abs(thermo_aged_spectral - optical_aged_spectral))
        
        return {
            'delta_e_thermo': delta_e_thermo,
            'delta_e_optical': delta_e_optical,
            'delta_e_between_methods': delta_e_between,
            'spectral_rmse': spectral_rmse.item(),
            'max_spectral_diff': max_diff.item(),
            'within_tolerance': delta_e_between < 2.0,
            'consistency_rating': 'Good' if delta_e_between < 2.0 else 
                                ('Moderate' if delta_e_between < 5.0 else 'Poor')
        }
    
    def monte_carlo_historical_check(self, pigment_indices, mixing_ratios=None, 
                               n_samples=100, historical_period="middle_tang"):
        """
        Perform Monte Carlo simulation to account for uncertainty in historical accuracy.
        
        Args:
            pigment_indices: Tensor/array of pigment indices
            mixing_ratios: Optional mixing ratios
            n_samples: Number of Monte Carlo samples
            historical_period: Target historical period
            
        Returns:
            Dictionary with statistical results of historical accuracy analysis
        """
        device = self.device
        pigment_indices = pigment_indices.to(device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(device)
        if isinstance(pigment_indices, torch.Tensor):
            indices = pigment_indices.detach().cpu().detach().numpy()
        else:
            indices = np.array(pigment_indices)
        if mixing_ratios is not None:
            if isinstance(mixing_ratios, torch.Tensor):
                ratios = mixing_ratios.detach().cpu().detach().numpy()
            else:
                ratios = np.array(mixing_ratios)
        else:
            ratios = None

        base_accuracy = self.analyze_art_historical_accuracy(
            pigment_indices,
            mixing_ratios,
            historical_period
        )

        accuracy_scores = []
        anachronistic_counts = []
        historically_avoided_counts = []
        period_appropriate_counts = []
        detailed_results = []

        for i in range(n_samples):
            perturbed_indices = self._perturb_pigment_indices(indices, historical_period)
            perturbed_tensor = torch.tensor(perturbed_indices, device=device)
            result = self.analyze_art_historical_accuracy(
                perturbed_tensor,
                mixing_ratios,
                historical_period
            )
            
            accuracy_scores.extend(result['historical_accuracy_scores'])
            anachronistic_count = sum(len(a) for a in result['anachronistic_pigments'])
            anachronistic_counts.append(anachronistic_count)
            appropriate_count = sum(len(p) for p in result['period_appropriate_pigments'])
            period_appropriate_counts.append(appropriate_count)
            avoided_count = len(result['inappropriate_combinations'])
            historically_avoided_counts.append(avoided_count)
            detailed_results.append(result)

        accuracy_scores = np.array(accuracy_scores)
        anachronistic_counts = np.array(anachronistic_counts)
        period_appropriate_counts = np.array(period_appropriate_counts)
        historically_avoided_counts = np.array(historically_avoided_counts)

        stats = {
            "historical_accuracy": {
                "mean": np.mean(accuracy_scores),
                "std": np.std(accuracy_scores),
                "min": np.min(accuracy_scores),
                "max": np.max(accuracy_scores),
                "q05": np.percentile(accuracy_scores, 5),
                "q50": np.percentile(accuracy_scores, 50),
                "q95": np.percentile(accuracy_scores, 95),
                "confidence_interval_95": (
                    np.percentile(accuracy_scores, 2.5),
                    np.percentile(accuracy_scores, 97.5)
                )
            },
            "anachronistic_pigments": {
                "mean": np.mean(anachronistic_counts),
                "std": np.std(anachronistic_counts),
                "min": np.min(anachronistic_counts),
                "max": np.max(anachronistic_counts)
            },
            "period_appropriate_pigments": {
                "mean": np.mean(period_appropriate_counts),
                "std": np.std(period_appropriate_counts),
                "min": np.min(period_appropriate_counts),
                "max": np.max(period_appropriate_counts)
            },
            "historically_avoided_combinations": {
                "mean": np.mean(historically_avoided_counts),
                "std": np.std(historically_avoided_counts),
                "min": np.min(historically_avoided_counts),
                "max": np.max(historically_avoided_counts)
            },
            "sample_count": n_samples,
            "baseline_result": base_accuracy,
            "historical_period": historical_period,
            "simulation_results": detailed_results
        }

        accuracy_threshold = 0.7 
        probability_accurate = np.mean(accuracy_scores >= accuracy_threshold) * 100
        stats["probability_historically_accurate"] = probability_accurate

        if probability_accurate > 90:
            confidence_rating = "Very High"
        elif probability_accurate > 75:
            confidence_rating = "High"
        elif probability_accurate > 50:
            confidence_rating = "Moderate"
        elif probability_accurate > 25:
            confidence_rating = "Low"
        else:
            confidence_rating = "Very Low"
            
        stats["historical_confidence_rating"] = confidence_rating

        most_common_anachronistic = self._identify_most_common_anachronistic(detailed_results)
        stats["most_problematic_pigments"] = most_common_anachronistic
        
        return stats

    def _perturb_pigment_indices(self, pigment_indices, historical_period="middle_tang"):
        """
        Perturb pigment indices to simulate historical uncertainty.
        
        Args:
            pigment_indices: Array of pigment indices
            historical_period: Target historical period
            
        Returns:
            Perturbed pigment indices
        """
        if isinstance(pigment_indices, list):
            perturbed = pigment_indices.copy()
        else:
            perturbed = pigment_indices.copy()
        if len(np.array(perturbed).shape) > 1:
            original_shape = np.array(perturbed).shape
            perturbed = np.array(perturbed).flatten().tolist()
        else:
            original_shape = None

        try:
            historical_data = self.validator.historical_data.get(historical_period, {})
            compatible_groups = historical_data.get('compatible_groups', [])

            period_pigments = []
            for group in compatible_groups:
                period_pigments.extend(group)
        except (AttributeError, KeyError):
            period_pigments = list(range(30)) 
        
        # Perturbation logic:
        # 1. Randomly replace anachronistic pigments with period-appropriate ones (15% chance)
        # 2. Randomly substitute similar pigments within the same color group (10% chance)
        # 3. Introduce small chance of completely random period pigment (5% chance)
        color_groups = {
            "white": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "red": [10, 11, 12, 14, 15, 16, 32],
            "blue": [17, 18, 19, 30],
            "green": [21, 22],
            "yellow": [23, 24, 26, 31, 33],
            "black": [25],
            "brown": [27],
            "gold": [28]
        }

        pigment_to_color = {}
        for color, pigments in color_groups.items():
            for p in pigments:
                pigment_to_color[p] = color

        for i in range(len(perturbed)):
            pigment_id = perturbed[i]
            if pigment_id not in period_pigments:
                if np.random.random() < 0.15:
                    color = pigment_to_color.get(pigment_id)
                    if color:
                        appropriate_options = [p for p in period_pigments 
                                            if p in color_groups.get(color, [])]
                        if appropriate_options:
                            perturbed[i] = np.random.choice(appropriate_options)
            else:
                if np.random.random() < 0.10:
                    color = pigment_to_color.get(pigment_id)
                    if color:
                        similar_options = [p for p in period_pigments 
                                        if p in color_groups.get(color, []) and p != pigment_id]
                        if similar_options:
                            perturbed[i] = np.random.choice(similar_options)
            
            if np.random.random() < 0.05 and period_pigments:
                perturbed[i] = np.random.choice(period_pigments)

        if original_shape is not None and len(original_shape) > 1:
            perturbed = np.array(perturbed).reshape(original_shape).tolist()
            
        return perturbed

    def _identify_most_common_anachronistic(self, results):
        """
        Identify the most commonly occurring anachronistic pigments.
        
        Args:
            results: List of historical accuracy analysis results
            
        Returns:
            List of most problematic pigments with occurrence counts
        """
        pigment_counts = {}
        
        for result in results:
            for batch_anachronistic in result['anachronistic_pigments']:
                for pigment_id in batch_anachronistic:
                    if pigment_id not in pigment_counts:
                        pigment_counts[pigment_id] = 0
                    pigment_counts[pigment_id] += 1
        
        sorted_pigments = sorted(pigment_counts.items(), key=lambda x: x[1], reverse=True)
        
        problematic_pigments = []
        for pigment_id, count in sorted_pigments:
            name = self.validator.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
            occurrence_percentage = (count / len(results)) * 100

            alternatives = []
            try:
                for suggestion in result.get('recommendations', []):
                    for rec in suggestion:
                        if rec.get('problematic_pigment') == name:
                            alternatives = rec.get('alternatives', [])
                            break
                    if alternatives:
                        break
            except:
                pass
            
            problematic_pigments.append({
                "id": pigment_id,
                "name": name,
                "occurrence_count": count,
                "occurrence_percentage": occurrence_percentage,
                "suggested_alternatives": alternatives
            })
        
        return problematic_pigments
    
    def analyze_binder_diffusion(self, layer_binders, layer_pigments, environmental_conditions=None):
        """
        Analyze potential binder diffusion and migration between layers.
        
        Args:
            layer_binders: List of binder types for each layer
            layer_pigments: List of lists of pigment indices for each layer
            environmental_conditions: Optional dict with environmental factors
            
        Returns:
            List of detected binder diffusion issues with severity scores
        """
        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "temperature": 298,
                "aging_years": 50
            }
        
        def get_binder_properties(binder_type):
            return {
                "animal_glue": {"viscosity": 0.7, "drying_time": 0.3, "polarity": "hydrophilic"},
                "gum_arabic": {"viscosity": 0.5, "drying_time": 0.4, "polarity": "hydrophilic"},
                "egg_tempera": {"viscosity": 0.6, "drying_time": 0.5, "polarity": "amphiphilic"},
                "drying_oil": {"viscosity": 0.8, "drying_time": 0.8, "polarity": "hydrophobic"}
            }.get(binder_type, {"viscosity": 0.6, "drying_time": 0.5, "polarity": "unknown"})
        
        diffusion_issues = []
        for i in range(len(layer_binders) - 1):
            upper_binder = layer_binders[i]
            lower_binder = layer_binders[i+1]
            upper_pigments = layer_pigments[i]
            lower_pigments = layer_pigments[i+1]
            upper_props = get_binder_properties(upper_binder)
            lower_props = get_binder_properties(lower_binder)

            if isinstance(upper_pigments, torch.Tensor):
                upper_pigments = upper_pigments.detach().cpu().detach().numpy().tolist()
            if isinstance(lower_pigments, torch.Tensor):
                lower_pigments = lower_pigments.detach().cpu().detach().numpy().tolist()
            if not isinstance(upper_pigments, list):
                upper_pigments = [upper_pigments]
            if not isinstance(lower_pigments, list):
                lower_pigments = [lower_pigments]
            
            # 1. Check for oil binder migration into water-based layers
            if upper_binder == "drying_oil" and lower_binder in ["animal_glue", "gum_arabic"]:
                diffusion_issues.append({
                    "type": "oil_penetration",
                    "upper_layer_index": i,
                    "lower_layer_index": i+1,
                    "severity": 0.7,
                    "description": f"Oil binder from upper layer may penetrate into the {lower_binder} layer below, potentially causing darkening and affecting adhesion."
                })
                
            # 2. Check for disruption of partially-dried layers
            if upper_props["drying_time"] > 0.6 and lower_props["drying_time"] < 0.5:
                diffusion_issues.append({
                    "type": "disrupted_drying",
                    "upper_layer_index": i,
                    "lower_layer_index": i+1,
                    "severity": 0.5,
                    "description": f"Slow-drying {upper_binder} over fast-drying {lower_binder} can lead to cracking and delamination as the lower layer is sealed before completely dry."
                })
                
            # 3. Check for pigment-binder interactions that affect diffusion
            porous_pigments = [0, 1, 2, 18]  # Kaolin, calcium carbonate, gypsum, lapis lazuli
            if any(p in porous_pigments for p in lower_pigments):
                if upper_binder == "drying_oil":
                    diffusion_issues.append({
                        "type": "enhanced_absorption",
                        "upper_layer_index": i,
                        "lower_layer_index": i+1,
                        "severity": 0.6,
                        "description": "Porous pigments in lower layer can draw excessive binder from the upper layer, leading to oil staining and potential adhesion issues."
                    })
                    
            # 4. Check for humidity effects on binder diffusion
            humidity = environmental_conditions.get("humidity", 0.5)
            if humidity > 0.7:
                if upper_binder in ["animal_glue", "gum_arabic"] or lower_binder in ["animal_glue", "gum_arabic"]:
                    diffusion_issues.append({
                        "type": "humidity_enhanced_diffusion",
                        "upper_layer_index": i,
                        "lower_layer_index": i+1,
                        "severity": humidity * 0.6,
                        "description": "High humidity can increase migration of water-soluble components between layers, potentially altering binding strength and pigment distribution."
                    })
                    
            # 5. Check for long-term diffusion due to aging
            aging_years = environmental_conditions.get("aging_years", 50)
            if aging_years > 30:
                age_factor = min(1.0, aging_years / 100)
                if "drying_oil" in [upper_binder, lower_binder]:
                    diffusion_issues.append({
                        "type": "long_term_oil_migration",
                        "upper_layer_index": i,
                        "lower_layer_index": i+1,
                        "severity": 0.3 * age_factor,
                        "description": "Over long time periods, components from drying oils can migrate between layers, causing yellowing and potential embrittlement."
                    })
                    
            # 6. Check for polarity mismatches
            if upper_props["polarity"] != lower_props["polarity"] and upper_props["polarity"] != "amphiphilic" and lower_props["polarity"] != "amphiphilic":
                diffusion_issues.append({
                    "type": "polarity_mismatch",
                    "upper_layer_index": i,
                    "lower_layer_index": i+1,
                    "severity": 0.4,
                    "description": f"Mismatch between {upper_props['polarity']} and {lower_props['polarity']} binders can lead to poor adhesion and separation of layers over time."
                })
        
        return diffusion_issues

    def simulate_full_rendering(self,
                             pigment_indices: torch.Tensor,
                             mixing_ratios: Optional[torch.Tensor] = None,
                             age_years: int = 0,
                             viewing_angle: float = 0.0,
                             light_direction: Optional[torch.Tensor] = None,
                             env_conditions: Optional[Dict] = None) -> Dict:
        """
        Perform complete physically-based rendering simulation with all effects.
        
        Args:
            pigment_indices: Indices of pigments [batch_size, n_pigments]
            mixing_ratios: Optional mixing ratios [batch_size, n_pigments]
            age_years: Optional aging simulation in years
            viewing_angle: Viewing angle in radians
            light_direction: Light direction vector
            env_conditions: Environmental conditions dictionary
            
        Returns:
            Dictionary with comprehensive rendering results, including:
            - rgb: RGB color values
            - spectral: Full spectral reflectance data
            - brdf_params: BRDF parameters for rendering
            - view_dependent_rgb: View-dependent color values
            - aging_info: Aging simulation results (if age_years > 0)
        """
        pigment_indices = pigment_indices.to(self.device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(self.device)
        if light_direction is not None:
            light_direction = light_direction.to(self.device)
        if env_conditions is None:
            env_conditions = {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3
            }

        render_result = self.optical_props.simulate_full_rendering(
            pigment_indices,
            mixing_ratios,
            age_years,
            viewing_angle,
            light_direction,
            env_conditions,
            self.validator
        )
        
        return render_result
    
    def analyze_layered_structure(self,
                               layer_pigments: List[torch.Tensor],
                               layer_binders: Optional[List[str]] = None,
                               layer_order: Optional[List[int]] = None,
                               years: int = 0,
                               environmental_conditions: Optional[Dict] = None) -> Dict:
        """
        Analyze a multi-layer paint structure with interactions between layers.
        
        Args:
            layer_pigments: List of tensors containing pigment indices for each layer
            layer_binders: Optional list of binder types for each layer
            layer_order: Optional specific ordering of layers (bottom to top)
            years: Optional simulation time in years for aging
            environmental_conditions: Optional dict with environmental factors
            
        Returns:
            Dictionary with layered structure analysis, including:
            - layer_stability: Individual stability of each layer
            - interlayer_issues: Issues between adjacent layers
            - overall_stability: Overall stability assessment
            - optical_properties: Optical properties of the complete structure
            - aging_effects: Aging simulation results (if years > 0)
        """
        layer_pigments = [layer.to(self.device) for layer in layer_pigments]

        if layer_binders is None:
            layer_binders = ["animal_glue"] * len(layer_pigments)
        if layer_order is None:
            layer_order = list(range(len(layer_pigments)))
        if environmental_conditions is None:
            environmental_conditions = {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3
            }

        layer_analysis = self.validator.analyze_painted_layers(
            layer_pigments,
            layer_binders,
            layer_order
        )

        layered_config = []
        for i, order_idx in enumerate(layer_order):
            layer_idx = layer_order.index(i)
            pigments = layer_pigments[layer_idx]
            
            if len(pigments.shape) > 1:
                n_pigments = pigments.shape[-1]
                ratios = torch.ones(1, n_pigments, device=self.device) / n_pigments
            else:
                ratios = None

            layered_config.append({
                'pigment_indices': pigments,
                'mixing_ratios': ratios,
                'thickness': 20.0, 
                'binder': layer_binders[layer_idx]
            })
            
        optical_props = self.calculate_optical_properties(
            pigment_indices=layer_pigments[0], 
            layered_configuration=layered_config,
            age_years=years
        )

        layer_analysis['optical_properties'] = optical_props

        if years > 0:
            layer_aging = []
            for i, pigments in enumerate(layer_pigments):
                if len(pigments.shape) > 1:
                    n_pigments = pigments.shape[-1]
                    ratios = torch.ones(1, n_pigments, device=self.device) / n_pigments
                else:
                    ratios = None

                aging_result = self.simulate_aging(
                    pigments,
                    ratios,
                    years,
                    environmental_conditions
                )
                
                layer_aging.append({
                    'layer_index': i,
                    'aging_effects': aging_result
                })
                
            layer_analysis['aging_effects'] = layer_aging
            
        return layer_analysis
    
    def analyze_art_historical_accuracy(self,
                                     pigment_indices: torch.Tensor,
                                     mixing_ratios: Optional[torch.Tensor] = None,
                                     historical_period: str = 'middle_tang',
                                     region: str = 'dunhuang') -> Dict:
        """
        Analyze the historical accuracy of a pigment selection.
        
        Args:
            pigment_indices: Indices of pigments [batch_size, n_pigments]
            mixing_ratios: Optional mixing ratios [batch_size, n_pigments]
            historical_period: Target historical period for comparison
            region: Geographical/cultural region for comparison
            
        Returns:
            Dictionary with historical accuracy analysis, including:
            - historical_accuracy_score: Overall accuracy score (0-1)
            - period_appropriate_pigments: Pigments appropriate for the period
            - anachronistic_pigments: Pigments not used in the period
            - compatibility: Historical compatibility assessment
            - recommendations: Suggestions for more accurate alternatives
        """
        pigment_indices = pigment_indices.to(self.device)
        if mixing_ratios is not None:
            mixing_ratios = mixing_ratios.to(self.device)
        if not hasattr(self.validator, 'historical_data'):
            raise ValueError("ThermodynamicValidator does not have historical data initialized")
            
        historical_data = self.validator.historical_data
        if historical_period not in historical_data:
            available_periods = list(historical_data.keys())
            historical_period = available_periods[0] 
            warnings.warn(f"Requested period not found. Using {historical_period} instead.")
            
        period_data = historical_data[historical_period]
        
        all_compatible_pigments = []
        for group in period_data['compatible_groups']:
            all_compatible_pigments.extend(group)
        
        batch_size = pigment_indices.shape[0]
        period_appropriate = []
        anachronistic = []
        
        for b in range(batch_size):
            sample_appropriate = []
            sample_anachronistic = []
            
            if len(pigment_indices.shape) > 1:
                sample_pigments = pigment_indices[b]
            else:
                sample_pigments = pigment_indices[b:b+1]
                
            for p in sample_pigments:
                p_id = p.item()
                
                if p_id in all_compatible_pigments:
                    sample_appropriate.append(p_id)
                else:
                    sample_anachronistic.append(p_id)
                    
            period_appropriate.append(sample_appropriate)
            anachronistic.append(sample_anachronistic)

        accuracy_scores = []
        for b in range(batch_size):
            if len(pigment_indices.shape) > 1:
                n_total = len(pigment_indices[b])
            else:
                n_total = 1
                
            n_appropriate = len(period_appropriate[b])
            accuracy = n_appropriate / n_total if n_total > 0 else 0
            accuracy_scores.append(accuracy)

        inappropriate_combinations = []
        for avoided in period_data['avoided_combinations']:
            for b in range(batch_size):
                if len(pigment_indices.shape) > 1:
                    sample_pigments = pigment_indices[b].cpu().detach().numpy()
                else:
                    sample_pigments = [pigment_indices[b].item()]
                    
                if avoided[0] in sample_pigments and avoided[1] in sample_pigments:
                    inappropriate_combinations.append({
                        'batch_idx': b,
                        'combination': (
                            self.pigment_db.id_to_name.get(avoided[0], f"Pigment {avoided[0]}"),
                            self.pigment_db.id_to_name.get(avoided[1], f"Pigment {avoided[1]}")
                        ),
                        'reason': "Historically avoided combination"
                    })

        recommendations = []
        for b in range(batch_size):
            batch_recommendations = []
            
            for p_id in anachronistic[b]:
                p_name = self.pigment_db.id_to_name.get(p_id, f"Pigment {p_id}")
                p_color = None
                for pigment in self.pigment_db.pigments:
                    if self.pigment_db.name_to_id.get(pigment["name"]) == p_id:
                        p_color = pigment["color"]
                        break
                
                if p_color:
                    alternatives = []
                    for alt_id in all_compatible_pigments:
                        for pigment in self.pigment_db.pigments:
                            if (self.pigment_db.name_to_id.get(pigment["name"]) == alt_id and 
                                pigment["color"] == p_color):
                                alternatives.append(alt_id)
                                
                    if alternatives:
                        batch_recommendations.append({
                            'anachronistic_pigment': p_name,
                            'color': p_color,
                            'period_appropriate_alternatives': [
                                self.pigment_db.id_to_name.get(alt_id, f"Pigment {alt_id}")
                                for alt_id in alternatives
                            ]
                        })
                        
            recommendations.append(batch_recommendations)
            
        return {
            'historical_accuracy_scores': accuracy_scores,
            'period_appropriate_pigments': period_appropriate,
            'anachronistic_pigments': anachronistic,
            'inappropriate_combinations': inappropriate_combinations,
            'recommendations': recommendations,
            'historical_period': historical_period
        }
    
    def visualize_results(self, 
                      result_data: Dict, 
                      result_type: str,
                      save_path: Optional[str] = None,
                      show: bool = True) -> None:
        """
        Visualize analysis results with appropriate plots.
        
        Args:
            result_data: Result dictionary from analysis functions
            result_type: Type of result to visualize 
                        ('pigment_analysis', 'stability', 'aging', 'optical', 'historical')
            save_path: Optional path to save visualization
            show: Whether to display the visualization
            
        Returns:
            None (displays or saves visualization)
        """
        plt.figure(figsize=(12, 8))
        
        if result_type == 'pigment_analysis':
            if 'pigment_probabilities' in result_data:
                probs = result_data['pigment_probabilities'][0].cpu().detach().numpy()
                top_indices = np.argsort(probs)[-10:][::-1] 
                
                plt.subplot(2, 2, 1)
                plt.barh([self.pigment_db.id_to_name.get(i, f"Pigment {i}") for i in top_indices], 
                         probs[top_indices])
                plt.title('Top 10 Pigment Probabilities')
                plt.xlabel('Probability')

                if 'color_clusters' in result_data:
                    clusters = result_data['color_clusters'][0]
                    
                    plt.subplot(2, 2, 2)
                    colors = [cluster['center'] for cluster in clusters]
                    percentages = [cluster['percentage'] for cluster in clusters]
                    
                    plt.barh(range(len(colors)), percentages, color=colors)
                    plt.title('Color Clusters')
                    plt.xlabel('Percentage')
                    plt.yticks([])

                if 'spectral_data' in result_data:
                    plt.subplot(2, 2, 3)
                    spectral = result_data['spectral_data'].cpu().detach().numpy()
                    plt.plot(np.linspace(400, 700, len(spectral)), spectral)
                    plt.title('Spectral Reflectance')
                    plt.xlabel('Wavelength (nm)')
                    plt.ylabel('Reflectance')

                if 'pigment_suggestions' in result_data:
                    plt.subplot(2, 2, 4)
                    suggestions = result_data['pigment_suggestions']
                    suggestion_names = [s['pigment_name'] for s in suggestions[:5]]
                    suggestion_probs = [s['probability'] for s in suggestions[:5]]
                    
                    plt.barh(suggestion_names, suggestion_probs)
                    plt.title('Top Pigment Suggestions')
                    plt.xlabel('Confidence')
                
        elif result_type == 'stability':
            if 'stability_score' in result_data:
                plt.subplot(2, 2, 1)
                plt.barh(['Stability Score'], [result_data['stability_score']])
                plt.title(f"Stability: {result_data['stability_rating']}")
                plt.xlim(0, 1)

                if 'detailed_issues' in result_data:
                    issues = result_data['detailed_issues']
                    issue_types = {}
                    
                    for issue in issues:
                        issue_type = issue['type']
                        impact = issue.get('impact', 0)
                        
                        if issue_type not in issue_types:
                            issue_types[issue_type] = 0
                        issue_types[issue_type] += impact
                        
                    if issue_types:
                        plt.subplot(2, 2, 2)
                        plt.barh(list(issue_types.keys()), list(issue_types.values()))
                        plt.title('Stability Issues')
                        plt.xlabel('Impact')

                if 'warnings' in result_data:
                    plt.subplot(2, 1, 2)
                    warnings_text = '\n'.join(result_data['warnings'])
                    plt.text(0.1, 0.5, warnings_text, wrap=True, fontsize=10)
                    plt.axis('off')
                    plt.title('Warnings')
                
        elif result_type == 'aging':
            if 'condition_score' in result_data:
                plt.subplot(2, 2, 1)
                plt.barh(['Condition After Aging'], [result_data['condition_score']])
                plt.title(f"Condition: {result_data['condition_rating']}")
                plt.xlim(0, 1)

                if 'aging_effects' in result_data:
                    effects = result_data['aging_effects']
                    
                    plt.subplot(2, 2, 2)
                    plt.barh(list(effects.keys()), list(effects.values()))
                    plt.title('Aging Effects')
                    plt.xlabel('Magnitude')

                if 'color_change' in result_data:
                    color_change = result_data['color_change']
                    
                    plt.subplot(2, 2, 3)
                    plt.bar(['Original', 'Aged'], [1, 1], color=[color_change['original_rgb'], color_change['aged_rgb']])
                    plt.title(f"Color Change (ΔE={color_change['delta_e']:.2f})")

                if 'spectral_aging' in result_data and 'original_spectral' in result_data['spectral_aging']:
                    plt.subplot(2, 2, 4)
                    original = result_data['spectral_aging']['original_spectral']
                    aged = result_data['spectral_aging']['aged_spectral']
                    wavelengths = result_data['spectral_aging']['wavelengths']
                    
                    plt.plot(wavelengths, original, label='Original')
                    plt.plot(wavelengths, aged, label='Aged')
                    plt.title('Spectral Changes')
                    plt.xlabel('Wavelength (nm)')
                    plt.ylabel('Reflectance')
                    plt.legend()
                
        elif result_type == 'optical':
            if 'rgb' in result_data:
                rgb = result_data['rgb'][0].cpu().detach().numpy()
                
                plt.subplot(2, 2, 1)
                plt.bar(['Red', 'Green', 'Blue'], rgb, color=['red', 'green', 'blue'])
                plt.title('RGB Values')
                plt.ylim(0, 1)

                plt.subplot(2, 2, 2)
                plt.imshow([[rgb]])
                plt.title('Color')
                plt.axis('off')

                if 'spectral' in result_data:
                    plt.subplot(2, 2, 3)
                    spectral = result_data['spectral'][0].cpu().detach().numpy()
                    plt.plot(np.linspace(400, 700, len(spectral)), spectral)
                    plt.title('Spectral Reflectance')
                    plt.xlabel('Wavelength (nm)')
                    plt.ylabel('Reflectance')

                if 'brdf_params' in result_data:
                    brdf = result_data['brdf_params']
                    params = {
                        'Roughness': brdf['roughness'].item() if torch.is_tensor(brdf['roughness']) else brdf['roughness'],
                        'Metallic': brdf['metallic'].item() if torch.is_tensor(brdf['metallic']) else brdf['metallic'],
                        'Specular': brdf['specular'].item() if torch.is_tensor(brdf['specular']) else brdf['specular'],
                        'Anisotropy': brdf['anisotropy'].item() if torch.is_tensor(brdf['anisotropy']) else brdf['anisotropy'],
                        'Subsurface': brdf['subsurface'].item() if torch.is_tensor(brdf['subsurface']) else brdf['subsurface']
                    }
                    
                    plt.subplot(2, 2, 4)
                    plt.barh(list(params.keys()), list(params.values()))
                    plt.title('BRDF Parameters')
                    plt.xlim(0, 1)
                
        elif result_type == 'historical':
            if 'historical_accuracy_scores' in result_data:
                plt.subplot(2, 2, 1)
                scores = result_data['historical_accuracy_scores']
                plt.barh(['Historical Accuracy'], [scores[0]])
                plt.title(f"Historical Accuracy ({result_data['historical_period']})")
                plt.xlim(0, 1)

                if 'period_appropriate_pigments' in result_data and 'anachronistic_pigments' in result_data:
                    appropriate = len(result_data['period_appropriate_pigments'][0])
                    anachronistic = len(result_data['anachronistic_pigments'][0])
                    
                    plt.subplot(2, 2, 2)
                    plt.pie([appropriate, anachronistic], 
                           labels=['Period-appropriate', 'Anachronistic'],
                           autopct='%1.1f%%')
                    plt.title('Pigment Distribution')

                if 'inappropriate_combinations' in result_data:
                    combinations = result_data['inappropriate_combinations']
                    
                    plt.subplot(2, 2, 3)
                    if combinations:
                        combo_text = '\n'.join([
                            f"{combo['combination'][0]} + {combo['combination'][1]}"
                            for combo in combinations
                        ])
                        plt.text(0.1, 0.5, combo_text, wrap=True, fontsize=10)
                        plt.title('Historically Avoided Combinations')
                    else:
                        plt.text(0.1, 0.5, "No inappropriate combinations found", fontsize=10)
                        plt.title('Historically Avoided Combinations')
                    plt.axis('off')

                if 'recommendations' in result_data:
                    recommendations = result_data['recommendations'][0]
                    
                    plt.subplot(2, 2, 4)
                    if recommendations:
                        rec_text = '\n'.join([
                            f"{rec['anachronistic_pigment']} → {', '.join(rec['period_appropriate_alternatives'][:2])}"
                            for rec in recommendations[:3]
                        ])
                        plt.text(0.1, 0.5, rec_text, wrap=True, fontsize=10)
                        plt.title('Recommended Alternatives')
                    else:
                        plt.text(0.1, 0.5, "No recommendations needed", fontsize=10)
                        plt.title('Recommended Alternatives')
                    plt.axis('off')
        
        else:
            plt.text(0.5, 0.5, f"Visualization not implemented for {result_type}", 
                    ha='center', fontsize=14)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def export_result_data(self, result_data: Dict, format_type: str = 'json', 
                  file_path: Optional[str] = None) -> Union[str, Dict]:
        """
        Export analysis results to various formats.
        
        Args:
            result_data: Result dictionary from analysis functions
            format_type: Export format ('json', 'csv', 'dict')
            file_path: Optional file path for saving the export
            
        Returns:
            Exported data as string or dictionary
        """
        def process_data(data):
            if isinstance(data, dict):
                return {k: process_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [process_data(item) for item in data]
            elif isinstance(data, tuple):
                return [process_data(item) for item in data]
            elif isinstance(data, torch.Tensor):
                return data.cpu().detach().numpy().tolist()
            elif isinstance(data, np.ndarray):
                return data.tolist()
            elif np.issubdtype(type(data), np.integer):
                return int(data)
            elif np.issubdtype(type(data), np.floating):
                return float(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            else:
                return data
                
        processed_data = process_data(result_data)
        
        if format_type == 'json':
            json_data = json.dumps(processed_data, indent=2)
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(json_data)
                return f"Data exported to {file_path}"
            else:
                return json_data
                
        elif format_type == 'csv':
            def flatten_dict(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key).items())
                    elif isinstance(v, list) and len(v) > 0 and not isinstance(v[0], dict):
                        items.append((new_key, str(v)))
                    elif not isinstance(v, list):
                        items.append((new_key, v))
                return dict(items)
                
            flat_data = flatten_dict(processed_data)
            output = io.StringIO() if not file_path else open(file_path, 'w', newline='')
            writer = csv.writer(output)
            writer.writerow(['Key', 'Value'])
            
            for key, value in flat_data.items():
                writer.writerow([key, value])
                
            if file_path:
                output.close()
                return f"Data exported to {file_path}"
            else:
                return output.getvalue()
                
        elif format_type == 'dict':
            return processed_data
            
        else:
            raise ValueError(f"Unsupported export format: {format_type}")