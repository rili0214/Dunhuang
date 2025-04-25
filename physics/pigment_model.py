import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pigment_database import DunhuangPigmentDB
from color_pig_map import ColorToPigmentMapper
from optical_properties import OpticalProperties
from themodynamicval import ThermodynamicValidator

# ===== Pigment Model with Realistic Physics =====
class PigmentModel(nn.Module):
    """
    Advanced pigment model for historically accurate and physically realistic
    simulation of Dunhuang mural pigments, integrating spectral rendering,
    multi-layer painting techniques, and physical aging simulation.
    """
    def __init__(
        self, 
        num_pigments=35,
        feature_dim=768,
        hidden_dim=256,
        max_pigments_per_layer=5,
        max_layers=3,
        spectral_bands=10,
        in_chans=3
    ):
        super().__init__()
        self.num_pigments = num_pigments
        self.feature_dim = feature_dim
        self.max_pigments_per_layer = max_pigments_per_layer
        self.max_layers = max_layers
        self.spectral_bands = spectral_bands
        self.in_chans = in_chans

        # Initialize pigment-to-feature projectors
        self.pigment_to_feature_projectors = nn.ModuleList([
            nn.Linear(num_pigments, feature_dim) 
            for _ in range(max_layers)
        ])
        
        # Initialize enhanced versions of component modules
        self.pigment_db = DunhuangPigmentDB()
        self.color_mapper = ColorToPigmentMapper(
            feature_dim, 128, num_pigments, 
            min_clusters=3, max_clusters=8
        )
        self.optical_properties = OpticalProperties(
            num_pigments, pigment_embedding_dim=64, spectral_bands=spectral_bands
        )
        self.thermodynamic_validator = ThermodynamicValidator(self.pigment_db)
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Layer structure predictor (predicts number of layers and their arrangement)
        self.layer_structure_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, max_layers),
            nn.Softmax(dim=-1)  # Probability distribution over possible layer counts
        )
        
        # Pigment selection per layer
        self.layer_pigment_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_pigments),
                nn.Softmax(dim=-1)
            ) for _ in range(max_layers)
        ])
        
        # Mixing ratios per layer
        self.layer_ratio_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim*2, hidden_dim),  
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, max_pigments_per_layer),
                nn.Softmax(dim=-1)
            ) for _ in range(max_layers)
        ])
        
        # Binder prediction per layer
        self.binder_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 binder types: animal_glue, gum_arabic, drying_oil, egg_tempera
            nn.Softmax(dim=-1)
        )
        
        # Age prediction (global for the painting)
        self.age_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 0-1 normalized age
        )
        
        # Environmental condition estimation
        self.environmental_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # humidity, light_exposure, air_pollutants, temperature_fluctuation
            nn.Sigmoid()
        )
        
        # Physical renderer with spectral rendering capabilities
        self.physical_renderer = nn.Sequential(
            # Input: Image + spectral reflectance + binder properties
            nn.Conv2d(in_chans + spectral_bands + 5, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_chans, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Layer blending and interaction model
        self.layer_interaction_model = nn.Sequential(
            nn.Conv2d(in_chans * self.max_layers, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_chans, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Refinement network for final adjustments
        self.refinement_network = nn.Sequential(
            nn.Conv2d(in_chans * 2, 64, kernel_size=3, padding=1),  # Input + rendered
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_chans, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Color histogram generator
        self.color_histogram_generator = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(16),  # 16x16 spatial resolution
            nn.Flatten(),
            nn.Linear(64*16*16, 3*256)  # 256 bins per RGB channel
        )
        
        # Binder type mapping
        self.binder_types = ["animal_glue", "gum_arabic", "drying_oil", "egg_tempera"]
        
        # Initialize feature adapter for various input dimensions
        self.feature_adapter = nn.Linear(3, feature_dim)
    
    def _extract_features(self, image):
        """Extract deep features from image"""
        # Process through backbone
        features = self.backbone(image)
        
        # Flatten spatial dimensions
        return features.reshape(features.size(0), -1)
    
    def _generate_color_histogram(self, image):
        """Generate color histogram from image"""
        return self.color_histogram_generator(image)
    
    def _adapt_features(self, features):
        """Adapt features to expected dimension"""
        if features.size(1) != self.feature_dim:
            if features.size(1) == 3: 
                adapted_features = self.feature_adapter(features)
            else:
                features_3d = features.unsqueeze(-1)
                pooled_features = F.adaptive_avg_pool1d(features_3d, self.feature_dim)
                adapted_features = pooled_features.squeeze(-1)
                # Make sure we return a 2D tensor [batch, feature_dim]
                print("Adapted features shape in AdaptFeatures function:", adapted_features.shape)
            return adapted_features
        return features
    
    def _predict_painting_structure(self, features):
        """Predict painting layer structure from image features"""
        # Predict layer probabilities
        layer_probs = self.layer_structure_predictor(features)
        
        # Determine number of layers (weighted average or highest prob)
        # For deterministic results, use argmax
        num_layers = torch.argmax(layer_probs, dim=1) + 1  # +1 because layers start from 1
        
        return {
            'layer_probs': layer_probs,
            'num_layers': num_layers
        }
    
    def _predict_layer_pigments(self, features, num_layers, pigment_analysis=None):
        """
        Predict pigments for each layer, considering ColorToPigmentMapper suggestions
        
        Args:
            features: Image features [batch_size, feature_dim]
            num_layers: Number of layers for each sample [batch_size]
            pigment_analysis: Optional results from ColorToPigmentMapper
            
        Returns:
            Dictionary with layer pigment data
        """
        batch_size = features.size(0)
        max_layers = min(num_layers.max().item(), self.max_layers)
        
        # Initialize storage for results
        layer_pigments = []
        layer_ratios = []
        
        # Process each layer
        for layer_idx in range(max_layers):
            # Get base pigment distribution from neural network
            base_pigment_probs = self.layer_pigment_predictor[layer_idx](features)
            
            # If we have ColorToPigmentMapper results, blend them with the base probabilities
            if pigment_analysis is not None:
                # Blend the probabilities (with weight toward ColorToPigmentMapper suggestions)
                blend_weight = 0.7  # 70% from ColorToPigmentMapper, 30% from base prediction
                
                for b in range(batch_size):
                    if 'pigment_probabilities' in pigment_analysis:
                        mapper_probs = pigment_analysis['pigment_probabilities'][b]
                        base_pigment_probs[b] = (1 - blend_weight) * base_pigment_probs[b] + blend_weight * mapper_probs
            
            # Select top-k pigments for this layer
            top_probs, top_indices = torch.topk(
                base_pigment_probs, self.max_pigments_per_layer, dim=1
            )
            
            # For layers beyond a sample's layer count, mask out
            layer_mask = (layer_idx < num_layers).float().unsqueeze(1)
            
            # Project pigment probabilities to match feature dimensions
            projected_pigments = self.pigment_to_feature_projectors[layer_idx](base_pigment_probs)
            
            # Create input for ratio prediction - just concatenate features and projected pigments
            ratio_input = torch.cat([features, projected_pigments], dim=1)
            
            # Predict mixing ratios for these pigments
            mixing_ratios = self.layer_ratio_predictor[layer_idx](ratio_input)
            
            # Apply mask for valid layers
            masked_indices = top_indices * layer_mask.long()
            masked_ratios = mixing_ratios * layer_mask
            
            layer_pigments.append(masked_indices)
            layer_ratios.append(masked_ratios)
        
        # Stack results
        layer_pigments = torch.stack(layer_pigments, dim=1)  # [batch, layers, max_pigments]
        layer_ratios = torch.stack(layer_ratios, dim=1)      # [batch, layers, max_pigments]
        
        return {
            'layer_pigments': layer_pigments,
            'layer_ratios': layer_ratios,
        }
    
    def _predict_binders_and_conditions(self, features):
        """Predict binder types and environmental conditions"""
        # Predict binder probabilities
        binder_probs = self.binder_predictor(features)
        
        # Get most likely binder for each sample
        binder_indices = torch.argmax(binder_probs, dim=1)
        
        # Convert to named binders
        binder_names = [self.binder_types[idx.item()] for idx in binder_indices]
        
        # Predict environmental conditions
        env_conditions = self.environmental_predictor(features)
        
        # Create dictionary of named conditions
        condition_names = ['humidity', 'light_exposure', 'air_pollutants', 'temperature_fluctuation']
        env_dict = {
            name: env_conditions[:, i] for i, name in enumerate(condition_names)
        }
        
        # Predict age factor
        age_factor = self.age_predictor(features).squeeze(-1)
        
        return {
            'binder_probs': binder_probs,
            'binder_indices': binder_indices,
            'binder_names': binder_names,
            'env_conditions': env_dict,
            'age_factor': age_factor
        }
    
    def _validate_mixture_stability(self, layer_data, binder_names, env_conditions):
        """Validate stability of pigment mixtures in each layer"""
        batch_size = layer_data['layer_pigments'].size(0)
        num_layers = layer_data['layer_pigments'].size(1)
        
        stability_results = []
        
        # Process each sample in batch
        for b in range(batch_size):
            layer_results = []
            
            # Process each layer
            for layer_idx in range(num_layers):
                # Get pigments and ratios for this layer
                pigments = layer_data['layer_pigments'][b, layer_idx]
                ratios = layer_data['layer_ratios'][b, layer_idx]
                
                # Skip if all zeros (inactive layer)
                if torch.all(pigments == 0):
                    continue
                
                # Validate stability
                stability = self.thermodynamic_validator.check_mixture_stability(
                    pigments.detach().cpu().numpy(),
                    ratios.detach().cpu().numpy(),
                    binder_type=binder_names[b],
                    environmental_conditions={
                        k: v[b].item() for k, v in env_conditions.items()
                    }
                )
                
                # If unstable, get recommendations
                if not stability["is_stable"]:
                    corrections = self.thermodynamic_validator.suggest_corrections(
                        pigments.detach().cpu().numpy(),
                        ratios.detach().cpu().numpy(),
                        binder_type=binder_names[b],
                        environmental_conditions={
                            k: v[b].item() for k, v in env_conditions.items()
                        }
                    )
                    
                    # Add correction suggestions
                    stability["corrections"] = corrections
                    
                    # Apply corrections to ratios if available
                    if "corrected_ratios" in corrections:
                        corrected = torch.tensor(
                            corrections["corrected_ratios"],
                            device=ratios.device,
                            dtype=ratios.dtype
                        )
                        
                        # Update the ratios in-place
                        layer_data['layer_ratios'][b, layer_idx] = corrected
                
                layer_results.append({
                    'layer_idx': layer_idx,
                    'stability': stability
                })
            
            # Also check multi-layer interactions
            layer_pigments = []
            for layer_idx in range(num_layers):
                if not torch.all(layer_data['layer_pigments'][b, layer_idx] == 0):
                    layer_pigments.append(
                        layer_data['layer_pigments'][b, layer_idx].detach().cpu().numpy().tolist()
                    )
            
            if len(layer_pigments) > 1:
                layer_analysis = self.thermodynamic_validator.analyze_painted_layers(
                    layer_pigments,
                    [binder_names[b]] * len(layer_pigments)
                )
                
                stability_results.append({
                    'layer_results': layer_results,
                    'layer_interaction_analysis': layer_analysis
                })
            else:
                stability_results.append({
                    'layer_results': layer_results,
                    'layer_interaction_analysis': None
                })
        
        return {
            'stability_results': stability_results,
            'updated_layer_data': layer_data
        }
    
    def _render_pigment_layers(self, image, layer_data, age_factor, env_conditions, years=100):
        """Render each pigment layer with physically-based simulation"""
        batch_size = image.size(0)
        num_layers = layer_data['layer_pigments'].size(1)
        
        rendered_layers = []
        
        # Render each layer
        for layer_idx in range(num_layers):
            layer_pigments = []
            
            # Process each sample
            for b in range(batch_size):
                # Get pigments and ratios for this sample's layer
                pigments = layer_data['layer_pigments'][b, layer_idx]
                ratios = layer_data['layer_ratios'][b, layer_idx]
                
                # Skip if all zeros (inactive layer)
                if torch.all(pigments == 0):
                    # Create blank layer
                    blank = torch.zeros_like(image[b:b+1])
                    layer_pigments.append(blank)
                    continue
                
                # Step 1: Generate aging spectral data using ThermodynamicValidator
                aging_data = self.thermodynamic_validator.generate_spectral_aging_data(
                    pigments, 
                    ratios,
                    years=years,
                    spectral_bands=self.spectral_bands,
                    environmental_conditions={
                        k: v[b].item() for k, v in env_conditions.items()
                    }
                )
                
                # Step 2: Get optical properties without aging (base properties)
                optical_props = self.optical_properties(
                    pigments, 
                    ratios
                )
                
                # Step 3: Apply pre-computed aging to optical properties
                aged_spectral = aging_data['aged_spectral']
                if isinstance(aged_spectral, np.ndarray):
                    aged_spectral = torch.tensor(aged_spectral, device=pigments.device)
                
                # Get RGB with aging applied
                aged_rgb = self.optical_properties.apply_spectral_aging(
                    optical_props['spectral'],
                    aged_spectral
                )
                
                # Extract properties for rendering
                spectral = aged_spectral              # [spectral_bands]
                rgb = aged_rgb                        # [3]
                surface_props = optical_props['surface_props']  # [3]
                
                # Create rendering parameters
                render_params = torch.cat([
                    spectral.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(1, -1, image.size(2), image.size(3)),
                    surface_props.unsqueeze(-1).unsqueeze(-1).expand(1, -1, image.size(2), image.size(3)),
                    age_factor[b:b+1].view(1, 1, 1, 1).expand(1, -1, image.size(2), image.size(3))
                ], dim=1)
                
                # Create similarity map between this layer's color and the input image
                similarity = self._compute_color_similarity(
                    image[b:b+1],
                    rgb.view(1, 3, 1, 1)
                )
                
                # Initialize this layer with similarity-weighted image
                layer_init = image[b:b+1] * similarity
                
                # Concatenate for rendering
                render_input = torch.cat([layer_init, render_params], dim=1)
                
                # Apply physical renderer
                rendered = self.physical_renderer(render_input)
                
                layer_pigments.append(rendered)
            
            # Stack batch results for this layer
            if layer_pigments:
                rendered_layer = torch.cat(layer_pigments, dim=0)
                rendered_layers.append(rendered_layer)
        
        # Apply layer interactions model
        if rendered_layers:
            # Pad with blank layers if needed
            while len(rendered_layers) < self.max_layers:
                rendered_layers.append(torch.zeros_like(image))
            
            # Stack layers along channel dimension
            stacked_layers = torch.cat(rendered_layers, dim=1)
            
            # Process through layer interaction model
            composite = self.layer_interaction_model(stacked_layers)
        else:
            # Fallback if no valid layers
            composite = torch.zeros_like(image)
        
        # Apply final refinement
        refined = self.refinement_network(torch.cat([image, composite], dim=1))
        
        return {
            'rendered_layers': rendered_layers,
            'composite': composite,
            'refined': refined
        }
    
    def _compute_color_similarity(self, image, reference_color):
        """Compute color similarity using Gaussian RBF kernel"""
        # Calculate squared difference
        diff = torch.sum((image - reference_color)**2, dim=1, keepdim=True)
        
        # Apply RBF kernel with adaptive sigma
        sigma = 0.15  # Adjustable parameter for softness
        similarity = torch.exp(-diff / (2 * sigma**2))
        
        return similarity
    
    def _simulate_aging(self, layer_data, binder_names, env_conditions, years=100):
        """Simulate aging effects over specified time period"""
        batch_size = layer_data['layer_pigments'].size(0)
        
        aging_results = []
        
        # Process each sample
        for b in range(batch_size):
            layer_aging = []
            
            # Process each layer
            for layer_idx in range(layer_data['layer_pigments'].size(1)):
                # Get pigments and ratios
                pigments = layer_data['layer_pigments'][b, layer_idx]
                ratios = layer_data['layer_ratios'][b, layer_idx]
                
                # Skip if all zeros (inactive layer)
                if torch.all(pigments == 0):
                    continue
                
                # Simulate aging
                aging = self.thermodynamic_validator.simulate_aging_effects(
                    pigments.detach().cpu().numpy(),
                    ratios.detach().cpu().numpy(),
                    years=years,
                    environmental_conditions={
                        k: v[b].item() for k, v in env_conditions.items()
                    }
                )
                
                layer_aging.append({
                    'layer_idx': layer_idx,
                    'aging_simulation': aging
                })
            
            aging_results.append(layer_aging)
        
        return aging_results
    
    def forward(self, image=None, features=None, color_histogram=None):
        """
        Full pipeline for realistic physics-based pigment analysis and rendering
        
        Args:
            image: Optional input image [B, C, H, W]
            features: Optional pre-extracted features [B, D]
            color_histogram: Optional pre-computed color histogram
            
        Returns:
            Dictionary with comprehensive results
        """
        # Extract features if not provided
        if features is None and image is not None:
            features = self._extract_features(image)
        elif features is None:
            raise ValueError("Either image or features must be provided")
            
        # Adapt features to expected dimension
        features = self._adapt_features(features)
        
        # Generate color histogram if not provided
        if color_histogram is None and image is not None:
            color_histogram = self._generate_color_histogram(image)
        
        # Step 1: Use ColorToPigmentMapper to analyze image colors and get pigment suggestions
        pigment_analysis = self.color_mapper.analyze_image_pigments(image, features)
        
        # Step 2: Analyze painting structure
        structure = self._predict_painting_structure(features)
        
        # Step 3: Use pigment suggestions to inform layer prediction
        # Incorporate color_mapper results into layer prediction
        layer_data = self._predict_layer_pigments(features, structure['num_layers'], pigment_analysis)
        
        # Step 4: Predict binders and environmental conditions
        binder_env_data = self._predict_binders_and_conditions(features)
        
        # Step 5: Validate stability of pigment mixtures
        stability_data = self._validate_mixture_stability(
            layer_data,
            binder_env_data['binder_names'],
            binder_env_data['env_conditions']
        )
        
        # Update layer data with any stability corrections
        layer_data = stability_data['updated_layer_data']
        
        # Step 6: Simulate aging effects
        aging_simulation = self._simulate_aging(
            layer_data,
            binder_env_data['binder_names'],
            binder_env_data['env_conditions'],
            years=100  # Default simulation period
        )
        
        # Step 7: Render pigment layers with physical simulation (if image provided)
        if image is not None:
            rendering = self._render_pigment_layers(
                image,
                layer_data,
                binder_env_data['age_factor'],
                binder_env_data['env_conditions'],
                years=100
            )
        else:
            rendering = None
        
        # Compile results
        results = {
            'pigment_analysis': pigment_analysis,  # Add ColorToPigmentMapper results
            'structure': structure,
            'layer_data': layer_data,
            'binder_data': binder_env_data,
            'stability_data': stability_data,
            'aging_simulation': aging_simulation,
            'rendering': rendering
        }
        
        return results
    
    def analyze_image(self, image, simulation_years=100):
        """
        Comprehensive analysis of an image for pigment reconstruction
        
        Args:
            image: Input image [B, C, H, W]
            simulation_years: Years to simulate for aging effects
            
        Returns:
            Analysis results with human-readable explanations
        """
        # Run the main model
        results = self.forward(image)
        
        batch_size = image.size(0)
        analysis = []
        
        # Generate analysis for each image
        for b in range(batch_size):
            # Get number of layers
            num_layers = results['structure']['num_layers'][b].item()
            
            # Dominant pigments per layer
            layer_pigments = []
            for layer_idx in range(num_layers):
                if layer_idx < results['layer_data']['layer_pigments'].size(1):
                    pigments = results['layer_data']['layer_pigments'][b, layer_idx]
                    ratios = results['layer_data']['layer_ratios'][b, layer_idx]
                    
                    # Get pigment names
                    pigment_details = []
                    for i in range(min(3, len(pigments))):  # Top 3 pigments
                        if pigments[i].item() > 0:
                            pigment_id = pigments[i].item()
                            pigment_name = self.pigment_db.id_to_name.get(pigment_id, f"Pigment {pigment_id}")
                            pigment_details.append({
                                'name': pigment_name,
                                'ratio': ratios[i].item(),
                                'color': self._get_pigment_color(pigment_id)
                            })
                    
                    layer_pigments.append({
                        'layer': layer_idx + 1,
                        'pigments': pigment_details
                    })
            
            # Binder information
            binder = results['binder_data']['binder_names'][b]
            
            # Stability analysis
            if b < len(results['stability_data']['stability_results']):
                stability = results['stability_data']['stability_results'][b]
                stability_issues = []
                
                for layer_result in stability['layer_results']:
                    layer_idx = layer_result['layer_idx']
                    stability_data = layer_result['stability']
                    
                    if not stability_data.get('is_stable', True):
                        warnings = stability_data.get('warnings', [])
                        
                        for warning in warnings:
                            stability_issues.append({
                                'layer': layer_idx + 1,
                                'issue': warning,
                                'severity': 'High' if 'stability_score' in stability_data and stability_data['stability_score'] < 0.5 else 'Medium'
                            })
                
                if stability['layer_interaction_analysis']:
                    for issue in stability['layer_interaction_analysis'].get('interlayer_issues', []):
                        stability_issues.append({
                            'layer': f"{issue['bottom_layer']+1} & {issue['top_layer']+1}",
                            'issue': "Interlayer interaction issues detected",
                            'severity': 'Medium'
                        })
            else:
                stability_issues = []
            
            # Aging predictions
            aging_issues = []
            if b < len(results['aging_simulation']):
                aging_data = results['aging_simulation'][b]
                
                for layer_aging in aging_data:
                    layer_idx = layer_aging['layer_idx']
                    simulation = layer_aging['aging_simulation']
                    
                    # Only report significant issues
                    if simulation['condition_score'] < 0.6:
                        aging_issues.append({
                            'layer': layer_idx + 1,
                            'condition': simulation['condition_rating'],
                            'primary_effect': simulation['primary_effect'],
                            'recommendation': simulation['recommendations'][0] if simulation['recommendations'] else "No specific recommendations"
                        })
            
            # Rendering quality
            rendering_quality = "High" if results['rendering'] and torch.mean(results['rendering']['refined']) > 0 else "N/A"
            
            # Compile analysis
            analysis.append({
                'num_layers': num_layers,
                'pigment_analysis': layer_pigments,
                'binder': binder,
                'stability_issues': stability_issues,
                'aging_issues': aging_issues,
                'rendering_quality': rendering_quality,
                'overall_recommendations': self._generate_recommendations(
                    layer_pigments, stability_issues, aging_issues
                )
            })
        
        return {
            'detailed_results': results,
            'analysis': analysis
        }
    
    def _get_pigment_color(self, pigment_id):
        """Get color name for a pigment ID"""
        for pigment in self.pigment_db.pigments:
            if self.pigment_db.name_to_id.get(pigment["name"]) == pigment_id:
                return pigment["color"]
        return "unknown"
    
    def _generate_recommendations(self, layer_pigments, stability_issues, aging_issues):
        """Generate overall recommendations based on analysis"""
        recommendations = []
        
        # Check for stability issues
        if stability_issues:
            severe_issues = [issue for issue in stability_issues if issue['severity'] == 'High']
            if severe_issues:
                recommendations.append(
                    "Consider reformulating the pigment mixture to address severe stability issues."
                )
            else:
                recommendations.append(
                    "Monitor for potential pigment interactions over time."
                )
        
        # Check for aging issues
        if aging_issues:
            # Get unique recommendations
            aging_recs = set()
            for issue in aging_issues:
                if 'recommendation' in issue:
                    aging_recs.add(issue['recommendation'])
            
            for rec in aging_recs:
                recommendations.append(rec)
        
        # Add general recommendations if needed
        if not recommendations:
            recommendations.append(
                "Pigment mixture appears stable and suitable for long-term preservation."
            )
        
        return recommendations
    
    def generate_physics_report(self, analysis_results):
        """
        Generate comprehensive physics property report
        
        Args:
            analysis_results: Results from the analysis process
            
        Returns:
            Detailed report with physics properties and recommendations
        """
        # Extract key information
        pigment_analysis = analysis_results.get('pigment_analysis', {})
        layer_data = analysis_results.get('layer_data', {})
        stability_data = analysis_results.get('stability_data', {})
        aging_simulation = analysis_results.get('aging_simulation', [])
        
        # Create comprehensive report
        report = {
            'title': "Dunhuang Mural Physics Analysis Report",
            'sections': []
        }
        
        # Section 1: Pigment Composition
        pigment_section = {
            'title': "Pigment Composition Analysis",
            'content': "The image contains the following pigments based on historical Dunhuang mural practices:",
            'subsections': []
        }
        
        if 'pigment_suggestions' in pigment_analysis:
            suggestions = pigment_analysis['pigment_suggestions']
            for suggestion in suggestions[:5]:  # Top 5 suggestions
                subsection = {
                    'title': f"{suggestion['pigment_name']} ({suggestion['probability']:.2f})",
                    'content': suggestion['explanation'],
                    'properties': {
                        'chemical_formula': self._get_pigment_formula(suggestion['pigment_id']),
                        'color': self._get_pigment_color(suggestion['pigment_id']),
                        'particle_size': self._get_pigment_particle_size(suggestion['pigment_id'])
                    }
                }
                pigment_section['subsections'].append(subsection)
        report['sections'].append(pigment_section)
        
        # Section 2: Layer Structure
        layer_section = {
            'title': "Layer Structure Analysis",
            'content': "The painting consists of the following layers:",
            'subsections': []
        }
        
        # Add layer details
        for b in range(len(analysis_results.get('analysis', []))):
            analysis = analysis_results['analysis'][b]
            for layer_info in analysis['pigment_analysis']:
                layer_num = layer_info['layer']
                layer_pigments = layer_info['pigments']
                
                subsection = {
                    'title': f"Layer {layer_num}",
                    'content': f"Contains {len(layer_pigments)} primary pigments",
                    'pigments': [p['name'] for p in layer_pigments]
                }
                layer_section['subsections'].append(subsection)
        report['sections'].append(layer_section)
        
        # Section 3: Stability Analysis
        stability_section = {
            'title': "Physical Stability Analysis",
            'content': "The following stability issues were identified:",
            'subsections': []
        }
        
        # Add stability issues
        for b in range(len(analysis_results.get('analysis', []))):
            analysis = analysis_results['analysis'][b]
            
            if not analysis['stability_issues']:
                stability_section['content'] = "No significant stability issues detected."
            else:
                for issue in analysis['stability_issues']:
                    subsection = {
                        'title': f"Layer {issue['layer']} Issue",
                        'content': issue['issue'],
                        'severity': issue['severity']
                    }
                    stability_section['subsections'].append(subsection)
        report['sections'].append(stability_section)
        
        # Section 4: Aging Prediction
        aging_section = {
            'title': "Aging Prediction (100 Years)",
            'content': "Based on physical simulation, the following aging effects are predicted:",
            'subsections': []
        }
        
        # Add aging predictions
        for b in range(len(analysis_results.get('analysis', []))):
            analysis = analysis_results['analysis'][b]
            
            if not analysis['aging_issues']:
                aging_section['content'] = "The pigment mixture is predicted to age well with minimal deterioration."
            else:
                for issue in analysis['aging_issues']:
                    subsection = {
                        'title': f"Layer {issue['layer']} Aging",
                        'content': f"Primary effect: {issue['primary_effect']}",
                        'recommendation': issue['recommendation']
                    }
                    aging_section['subsections'].append(subsection)
        report['sections'].append(aging_section)
        
        # Section 5: Preservation Recommendations
        recommendations_section = {
            'title': "Preservation Recommendations",
            'content': "Based on the analysis, the following preservation measures are recommended:",
            'recommendations': []
        }
        
        # Add recommendations
        for b in range(len(analysis_results.get('analysis', []))):
            analysis = analysis_results['analysis'][b]
            for rec in analysis['overall_recommendations']:
                if rec not in recommendations_section['recommendations']:
                    recommendations_section['recommendations'].append(rec)
        report['sections'].append(recommendations_section)
        
        return report

    def _get_pigment_formula(self, pigment_id):
        """Get chemical formula for a pigment"""
        for pigment in self.pigment_db.pigments:
            if self.pigment_db.name_to_id.get(pigment["name"]) == pigment_id:
                return pigment.get("formula", "Unknown")
        return "Unknown"

    def _get_pigment_particle_size(self, pigment_id):
        """Get particle size information for a pigment"""
        for pigment in self.pigment_db.pigments:
            if self.pigment_db.name_to_id.get(pigment["name"]) == pigment_id:
                if "particle_size" in pigment:
                    return f"{pigment['particle_size']['mean']} {pigment['particle_size'].get('unit', 'micron')}"
        return "Unknown"
    


if __name__ == "__main__":
    # Initialize the model
    model = PigmentModel(num_pigments=35, feature_dim=768, in_chans=3)
    
    # Set model to evaluation mode
    model.eval()

    def print_model_dimensions():
        print("\n=== Model Dimensions ===")
        # Check feature dimensions
        print(f"Feature dimension: {model.feature_dim}")
        
        # Check layer predictor input dimensions
        for i, layer in enumerate(model.layer_ratio_predictor):
            first_linear = layer[0]  # Get first layer (Linear)
            print(f"Layer {i} ratio predictor input dim: {first_linear.in_features}")
            
        # Check projector dimensions
        for i, projector in enumerate(model.pigment_to_feature_projectors):
            print(f"Layer {i} projector: in={projector.in_features}, out={projector.out_features}")

    def test_simplified():
        print("\n=== Simplified Test ===")
    
        # Create a simple test image
        test_image = torch.zeros(1, 3, 64, 64)
        test_image[0, 0, :32, :32] = 0.9  # Red in top-left
        
        with torch.no_grad():
            # Extract features
            features = model._extract_features(test_image)
            print(f"Features shape: {features.shape}")

            # Adapt features to expected dimension
            features = model._adapt_features(features)
            print(f"Adapted features shape: {features.shape}")
            
            # Test structure prediction
            structure = model._predict_painting_structure(features)
            print(f"Predicted layers: {structure['num_layers'].item()}")
            
            # Create dummy layer data to test other components
            dummy_layer_data = {
                'layer_pigments': torch.ones(1, 1, 5).long(),
                'layer_ratios': torch.ones(1, 1, 5) / 5
            }
            
            # Test binder prediction
            binder_data = model._predict_binders_and_conditions(features)
            print(f"Predicted binder: {binder_data['binder_names'][0]}")
            
            # Test stability validation
            try:
                stability = model._validate_mixture_stability(
                    dummy_layer_data,
                    binder_data['binder_names'],
                    binder_data['env_conditions']
                )
                print("Stability validation successful")
            except Exception as e:
                print(f"Stability validation error: {str(e)}")
            
            # Test aging simulation
            try:
                aging = model._simulate_aging(
                    dummy_layer_data,
                    binder_data['binder_names'],
                    binder_data['env_conditions']
                )
                print("Aging simulation successful")
            except Exception as e:
                print(f"Aging simulation error: {str(e)}")
        
        return "Simplified test completed"
    
    # Test 1: Basic functionality test
    def test_basic_functionality():
        print("\n=== Test 1: Basic Functionality ===")
        
        # Create a simple test image
        test_image = torch.zeros(1, 3, 64, 64)
        # Red area in top-left
        test_image[0, 0, :32, :32] = 0.9
        # Blue area in bottom-right
        test_image[0, 2, 32:, 32:] = 0.9
        
        # Run forward pass
        with torch.no_grad():
            results = model(test_image)
        
        # Print basic results
        print(f"Predicted number of layers: {results['structure']['num_layers'].item()}")
        
        if results['rendering'] is not None:
            print("Rendering completed successfully")
        
        # Verify structure
        assert 'structure' in results, "Should include structure information"
        assert 'layer_data' in results, "Should include layer data"
        assert 'binder_data' in results, "Should include binder information"
        assert 'stability_data' in results, "Should include stability analysis"
        assert 'aging_simulation' in results, "Should include aging simulation"
        
        # Display sample results
        num_layers = results['structure']['num_layers'][0].item()
        print(f"Predicting {num_layers} layers for this image")
        
        # Show dominant pigments in first layer
        if num_layers > 0 and results['layer_data']['layer_pigments'].size(1) > 0:
            pigments = results['layer_data']['layer_pigments'][0, 0]
            ratios = results['layer_data']['layer_ratios'][0, 0]
            
            print("\nTop pigments in first layer:")
            for i in range(min(3, len(pigments))):
                if pigments[i].item() > 0:
                    pigment_id = pigments[i].item()
                    pigment_name = model.pigment_db.id_to_name.get(int(pigment_id), f"Pigment {pigment_id}")
                    print(f"  - {pigment_name} (ratio: {ratios[i].item():.2f})")
        
        return results

    # Test 2: Comprehensive analysis test
    def test_comprehensive_analysis():
        print("\n=== Test 2: Comprehensive Analysis ===")
        
        # Create a more complex test image
        test_image = torch.zeros(1, 3, 64, 64)
        # Red area
        test_image[0, 0, :32, :32] = 0.9
        # Green area
        test_image[0, 1, :32, 32:] = 0.9
        # Blue area
        test_image[0, 2, 32:, :32] = 0.9
        # Yellow area (Red + Green)
        test_image[0, 0, 32:, 32:] = 0.9
        test_image[0, 1, 32:, 32:] = 0.9
        
        # Run comprehensive analysis
        with torch.no_grad():
            analysis = model.analyze_image(test_image)
        
        # Print analysis for first image
        if 'analysis' in analysis and len(analysis['analysis']) > 0:
            img_analysis = analysis['analysis'][0]
            
            print(f"Number of layers: {img_analysis['num_layers']}")
            print(f"Binder: {img_analysis['binder']}")
            
            print("\nPigment analysis:")
            for layer in img_analysis['pigment_analysis']:
                print(f"Layer {layer['layer']}:")
                for pigment in layer['pigments']:
                    print(f"  - {pigment['name']} ({pigment['color']}, ratio: {pigment['ratio']:.2f})")
            
            print("\nStability issues:")
            if img_analysis['stability_issues']:
                for issue in img_analysis['stability_issues']:
                    print(f"  - Layer {issue['layer']}: {issue['issue']} (Severity: {issue['severity']})")
            else:
                print("  None detected")
            
            print("\nAging issues:")
            if img_analysis['aging_issues']:
                for issue in img_analysis['aging_issues']:
                    print(f"  - Layer {issue['layer']}: {issue['condition']} - {issue['primary_effect']}")
                    print(f"    Recommendation: {issue['recommendation']}")
            else:
                print("  None detected")
            
            print("\nOverall recommendations:")
            for rec in img_analysis['overall_recommendations']:
                print(f"  - {rec}")
        
        return analysis

    # Test 3: Environmental effects test
    def test_environmental_effects():
        print("\n=== Test 3: Environmental Effects ===")
        
        # Create a simple test image
        test_image = torch.zeros(1, 3, 64, 64)
        # Add some color (uniform green)
        test_image[0, 1, :, :] = 0.9
        
        # Extract features from the image
        with torch.no_grad():
            features = model._extract_features(test_image)
        
        # Test with different environmental conditions
        env_combinations = [
            {"name": "Ideal conditions", "humidity": 0.1, "light_exposure": 0.1},
            {"name": "High humidity", "humidity": 0.9, "light_exposure": 0.1},
            {"name": "High light exposure", "humidity": 0.1, "light_exposure": 0.9},
            {"name": "Poor conditions", "humidity": 0.9, "light_exposure": 0.9}
        ]
        
        for env in env_combinations:
            print(f"\nTesting with {env['name']}:")
            
            # Override environmental predictions
            with torch.no_grad():
                # Run forward pass
                results = model(test_image, features)
                
                # Override environmental conditions
                for k, v in env.items():
                    if k != "name" and k in results['binder_data']['env_conditions']:
                        results['binder_data']['env_conditions'][k][0] = torch.tensor(v)
                
                # Regenerate stability analysis
                stability_data = model._validate_mixture_stability(
                    results['layer_data'],
                    results['binder_data']['binder_names'],
                    results['binder_data']['env_conditions']
                )
                
                # Simulate aging
                aging_simulation = model._simulate_aging(
                    results['layer_data'],
                    results['binder_data']['binder_names'],
                    results['binder_data']['env_conditions'],
                    years=100
                )
                
                # Print stability and aging results for first layer
                if (len(stability_data['stability_results']) > 0 and 
                    len(stability_data['stability_results'][0]['layer_results']) > 0):
                    
                    layer_stability = stability_data['stability_results'][0]['layer_results'][0]['stability']
                    print(f"Stability score: {layer_stability['stability_score']:.2f} ({layer_stability['stability_rating']})")
                    
                    if len(aging_simulation) > 0 and len(aging_simulation[0]) > 0:
                        layer_aging = aging_simulation[0][0]['aging_simulation']
                        print(f"Condition after aging: {layer_aging['condition_score']:.2f} ({layer_aging['condition_rating']})")
                        print(f"Primary aging effect: {layer_aging['primary_effect']}")
        
        return env_combinations

    # Test 4: Pigment substitution test
    def test_pigment_substitution():
        print("\n=== Test 4: Pigment Substitution ===")
        
        # Create test image with red color (cinnabar-like)
        test_image = torch.zeros(1, 3, 64, 64)
        test_image[0, 0, :, :] = 0.9  # High red channel
        
        # Run analysis
        with torch.no_grad():
            results = model(test_image)
        
        # Check if we detected unstable pigments
        has_unstable_mixture = False
        if 'stability_data' in results:
            stability_results = results['stability_data']['stability_results']
            if len(stability_results) > 0:
                for layer_result in stability_results[0]['layer_results']:
                    if not layer_result['stability'].get('is_stable', True):
                        has_unstable_mixture = True
                        
                        print(f"Found unstable mixture in layer {layer_result['layer_idx'] + 1}")
                        print(f"Stability score: {layer_result['stability']['stability_score']:.2f}")
                        
                        if 'corrections' in layer_result['stability']:
                            corrections = layer_result['stability']['corrections']
                            
                            print("\nCorrection suggestions:")
                            for suggestion in corrections['suggestions']:
                                if suggestion['type'] == "pigment_replacement" and 'alternatives' in suggestion:
                                    print(f"Suggested replacements for {suggestion['problematic_pigment']}:")
                                    
                                    for alt in suggestion['alternatives']:
                                        print(f"  - {alt['name']} (stability: {alt['stability_score']:.2f})")
                                        
                                elif suggestion['type'] == "ratio_adjustment" and 'pigment' in suggestion:
                                    print(f"Adjust ratio for {suggestion['pigment']}:")
                                    print(f"  From {suggestion['original_ratio']:.2f} to {suggestion['new_ratio']:.2f}")
        
        if not has_unstable_mixture:
            print("No unstable mixtures detected in this test")
        
        return results

    # Test 5: Aging simulation test
    def test_aging_simulation():
        print("\n=== Test 5: Aging Simulation ===")
        
        # Create a simple test image
        test_image = torch.zeros(1, 3, 64, 64)
        # Add some color (blue - like azurite)
        test_image[0, 2, :, :] = 0.9
        
        # Run aging analysis for different time periods
        time_periods = [10, 50, 100, 200, 500]
        
        for years in time_periods:
            with torch.no_grad():
                # Run forward pass
                results = model(test_image)
                
                # Run aging simulation
                aging_simulation = model._simulate_aging(
                    results['layer_data'],
                    results['binder_data']['binder_names'],
                    results['binder_data']['env_conditions'],
                    years=years
                )
                
                if len(aging_simulation) > 0 and len(aging_simulation[0]) > 0:
                    layer_aging = aging_simulation[0][0]['aging_simulation']
                    
                    print(f"\nAfter {years} years:")
                    print(f"Condition: {layer_aging['condition_score']:.2f} ({layer_aging['condition_rating']})")
                    print(f"Effects: darkening={layer_aging['aging_effects']['darkening']:.2f}, " +
                         f"yellowing={layer_aging['aging_effects']['yellowing']:.2f}, " +
                         f"fading={layer_aging['aging_effects']['fading']:.2f}, " +
                         f"cracking={layer_aging['aging_effects']['cracking']:.2f}")
                    print(f"Primary effect: {layer_aging['primary_effect']}")
                    print(f"Explanation: {layer_aging['explanation']}")
                else:
                    print(f"\nNo aging data for {years} years")
        
        return time_periods

    # Test 6: Example-based rendering test
    def test_rendering():
        print("\n=== Test 6: Rendering Test ===")
        
        # Try to load an example image if possible
        try:
            # Create a simple image with strong colors
            test_image = torch.zeros(1, 3, 128, 128)
            # Red circle in the middle
            for i in range(128):
                for j in range(128):
                    dist = ((i - 64) ** 2 + (j - 64) ** 2) ** 0.5
                    if dist < 32:
                        test_image[0, 0, i, j] = 0.9  # Red channel
            
            # Blue background with gradient
            for i in range(128):
                for j in range(128):
                    dist = ((i - 64) ** 2 + (j - 64) ** 2) ** 0.5
                    if dist >= 32:
                        test_image[0, 2, i, j] = 0.5 + 0.4 * (dist - 32) / 64  # Blue channel
            
            # Run rendering
            with torch.no_grad():
                results = model(test_image)
                
                if results['rendering']:
                    # Get rendered layers and composite
                    rendered_layers = results['rendering']['rendered_layers']
                    composite = results['rendering']['composite']
                    refined = results['rendering']['refined']
                    
                    print(f"Number of rendered layers: {len(rendered_layers)}")
                    
                    # Create a figure to show the results
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Show original image
                    axes[0].imshow(test_image[0].permute(1, 2, 0).cpu().numpy())
                    axes[0].set_title("Original Image")
                    axes[0].axis('off')
                    
                    # Show composite
                    axes[1].imshow(composite[0].permute(1, 2, 0).cpu().numpy())
                    axes[1].set_title("Composite Rendering")
                    axes[1].axis('off')
                    
                    # Show refined result
                    axes[2].imshow(refined[0].permute(1, 2, 0).cpu().numpy())
                    axes[2].set_title("Refined Rendering")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig("rendering_test.png")
                    print("Saved rendering visualization to 'rendering_test.png'")
                    plt.close()
                    
                    # Optional: Display individual layers
                    if len(rendered_layers) > 0:
                        fig, axes = plt.subplots(1, min(3, len(rendered_layers)), figsize=(15, 5))
                        if len(rendered_layers) == 1:
                            axes = [axes]  # Make iterable for single layer case
                            
                        for i, layer in enumerate(rendered_layers[:3]):  # Show first 3 layers
                            axes[i].imshow(layer[0].permute(1, 2, 0).cpu().numpy())
                            axes[i].set_title(f"Layer {i+1}")
                            axes[i].axis('off')
                            
                        plt.tight_layout()
                        plt.savefig("layer_visualization.png")
                        print("Saved layer visualization to 'layer_visualization.png'")
                        plt.close()
                else:
                    print("Rendering data not available")
                    
        except Exception as e:
            print(f"Error in rendering test: {str(e)}")
            
        return None  # No need to return the potentially large rendering data

    # Run the tests
    with torch.no_grad():  # Ensure we don't compute gradients for testing
        print_model_dimensions()
        test1 = test_simplified()
        test_results = test_basic_functionality()
        analysis_results = test_comprehensive_analysis()
        env_results = test_environmental_effects()
        substitution_results = test_pigment_substitution()
        aging_results = test_aging_simulation()
        rendering_results = test_rendering()
    
    print("\nAll tests completed!")