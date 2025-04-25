import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from pigment_database import DunhuangPigmentDB

# ===== Color to Pigment Mapper with Adaptive Clustering =====
class ColorToPigmentMapper(nn.Module):
    """
    Maps image color features to physically realistic pigment distributions based on
    Dunhuang mural research, with adaptive color clustering and spectral mixing.
    """
    def __init__(self, swin_feat_dim=768, color_feat_dim=128, num_pigments=35, 
                 min_clusters=3, max_clusters=8, cluster_threshold=0.02):
        super().__init__()
        self.num_pigments = num_pigments
        self.swin_feat_dim = swin_feat_dim
        self.color_feat_dim = color_feat_dim
        self.min_clusters = min_clusters  # Minimum number of color clusters to detect
        self.max_clusters = max_clusters  # Maximum number of color clusters to detect
        self.cluster_threshold = cluster_threshold  # Minimum cluster size as % of image
        
        # Initialize pigment database
        self.pigment_db = DunhuangPigmentDB()
        
        # Color feature extraction - improved with multi-scale processing
        self.color_encoder = nn.Sequential(
            nn.Linear(3*256, 512),  # RGB histogram (256 bins per channel)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, color_feat_dim)
        )
        
        # Add feature adaptation layer to handle varying input dimensions
        self.feature_adapter = nn.Linear(3, swin_feat_dim)  # Default simple adapter
        
        # Joint feature processing with spatial attention
        self.joint_mlp = nn.Sequential(
            nn.Linear(swin_feat_dim + color_feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_pigments)
        )
        
        # Per-pigment selection probability (based on historical frequency)
        # Trained on spectral data from actual Dunhuang murals
        self.pigment_prior = nn.Parameter(torch.ones(num_pigments))
        
        # Cultural/regional style modulation factors
        self.style_modulation = nn.Parameter(torch.ones(num_pigments))
        
        # Per-color spectral characteristics - more accurate than simple RGB centroids
        self.spectral_profiles = {
            'white': {
                'reflectance_peak': [0.92, 0.92, 0.92],
                'spectral_variance': 0.02,
                'possible_pigments': ['kaolin', 'calcium_carbonate', 'gypsum', 'lead_white',
                                     'quartz', 'lead_chloride', 'muscovite', 'talc', 
                                     'anhydrite', 'lead_sulfate'],
                'excluded_pigments': []
            },
            'red': {
                'reflectance_peak': [0.82, 0.12, 0.12],
                'spectral_variance': 0.12,
                'possible_pigments': ['cinnabar', 'vermilion', 'hematite', 'lead_oxide', 
                                     'realgar', 'cinnabar_variant', 'red_ochre', 'lac'],
                'excluded_pigments': ['gold_leaf', 'carbon_black']
            },
            'blue': {
                'reflectance_peak': [0.12, 0.25, 0.85],
                'spectral_variance': 0.08,
                'possible_pigments': ['azurite', 'lapis_lazuli', 'synthetic_blue', 'indigo'],
                'excluded_pigments': ['yellow_ochre', 'orpiment', 'gamboge']
            },
            'green': {
                'reflectance_peak': [0.12, 0.75, 0.22],
                'spectral_variance': 0.09,
                'possible_pigments': ['atacamite', 'malachite'],
                'excluded_pigments': ['cinnabar', 'vermilion', 'carbon_black']
            },
            'yellow': {
                'reflectance_peak': [0.88, 0.82, 0.12],
                'spectral_variance': 0.06,
                'possible_pigments': ['yellow_ochre', 'orpiment', 'arsenic_sulfide', 
                                     'gamboge', 'phellodendron'],
                'excluded_pigments': ['carbon_black', 'indigo']
            },
            'black': {
                'reflectance_peak': [0.05, 0.05, 0.05],
                'spectral_variance': 0.01,
                'possible_pigments': ['carbon_black'],
                'excluded_pigments': ['lead_white', 'calcium_carbonate', 'gold_leaf']
            },
            'brown': {
                'reflectance_peak': [0.45, 0.30, 0.18],
                'spectral_variance': 0.15,
                'possible_pigments': ['brown_alteration', 'hematite', 'yellow_ochre'],
                'excluded_pigments': ['synthetic_blue', 'lapis_lazuli']
            },
            'gold': {
                'reflectance_peak': [0.85, 0.70, 0.10],
                'spectral_variance': 0.03,
                'possible_pigments': ['gold_leaf', 'orpiment'],
                'excluded_pigments': ['carbon_black', 'indigo', 'azurite']
            },
            'gray': {
                'reflectance_peak': [0.50, 0.50, 0.50],
                'spectral_variance': 0.10,
                'possible_pigments': ['gray_mixture', 'carbon_black', 'lead_white'],
                'excluded_pigments': []
            },
            'purple': {
                'reflectance_peak': [0.52, 0.12, 0.52],
                'spectral_variance': 0.08,
                'possible_pigments': ['lac', 'indigo', 'cinnabar'],
                'excluded_pigments': ['yellow_ochre', 'orpiment']
            },
            'orange': {
                'reflectance_peak': [0.88, 0.45, 0.12],
                'spectral_variance': 0.10,
                'possible_pigments': ['realgar', 'lead_oxide', 'red_ochre', 'orpiment'],
                'excluded_pigments': ['azurite', 'lapis_lazuli', 'carbon_black']
            }
        }
        
        # Initialize name-to-id mapping for pigments
        self._init_pigment_name_map()
        
    def _init_pigment_name_map(self):
        """Initialize mapping from pigment names to their IDs"""
        self.pigment_name_to_id = {}
        for color, profile in self.spectral_profiles.items():
            for pigment_name in profile['possible_pigments']:
                if pigment_name in self.pigment_db.name_to_id:
                    self.pigment_name_to_id[pigment_name] = self.pigment_db.name_to_id[pigment_name]
                    
    def _adaptive_color_clustering(self, image_tensor):
        """
        Perform adaptive color clustering based on image complexity
        """
        # Convert to numpy for clustering
        if isinstance(image_tensor, torch.Tensor):
            # Handle different shapes - could be [B, C, H, W] or [B, C]
            if len(image_tensor.shape) == 4:
                # [B, C, H, W] -> [B, H*W, C]
                batch_size, channels, height, width = image_tensor.shape
                image_np = image_tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
                image_np = image_np.detach().cpu().numpy()
            elif len(image_tensor.shape) == 2:
                # [B, C] -> [B, 1, C]
                image_np = image_tensor.unsqueeze(1).detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")
        else:
            # Assume already numpy
            image_np = image_tensor
            
        batch_size = image_np.shape[0]
        all_clusters = []
        
        for b in range(batch_size):
            # For each image in batch
            img_data = image_np[b]
            
            if len(img_data.shape) == 3:  # [H, W, C]
                pixels = img_data.reshape(-1, img_data.shape[-1])
            else:  # [N, C]
                pixels = img_data
                
            # Sample pixels for efficiency if there are too many
            max_pixels = 5000
            if pixels.shape[0] > max_pixels:
                indices = np.random.choice(pixels.shape[0], max_pixels, replace=False)
                pixel_sample = pixels[indices]
            else:
                pixel_sample = pixels
                
            # Calculate optimal number of clusters using silhouette score
            optimal_k = self._find_optimal_clusters(pixel_sample)
            
            # Use KMeans clustering with the optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans.fit(pixel_sample)
            
            # Get cluster centers and sizes
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Calculate cluster sizes as percentages
            unique_labels, counts = np.unique(labels, return_counts=True)
            percentages = counts / counts.sum()
            
            # Filter out clusters that are too small
            significant_clusters = []
            for label, percentage in zip(unique_labels, percentages):
                if percentage >= self.cluster_threshold:
                    significant_clusters.append({
                        'center': centers[label],
                        'percentage': percentage
                    })
            
            # Sort by percentage (descending)
            significant_clusters.sort(key=lambda x: x['percentage'], reverse=True)
            
            # Map clusters to color names and add pigment constraints
            for cluster in significant_clusters:
                rgb = cluster['center']
                color_name = self._rgb_to_color_name(rgb)
                cluster['color_name'] = color_name
                
                # Add spectral profile information
                if color_name in self.spectral_profiles:
                    profile = self.spectral_profiles[color_name]
                    cluster['spectral_profile'] = profile
                    
                    # Convert pigment names to IDs
                    possible_pigments = []
                    for p_name in profile['possible_pigments']:
                        if p_name in self.pigment_name_to_id:
                            possible_pigments.append(self.pigment_name_to_id[p_name])
                    cluster['possible_pigments'] = possible_pigments
                    
                    excluded_pigments = []
                    for p_name in profile['excluded_pigments']:
                        if p_name in self.pigment_name_to_id:
                            excluded_pigments.append(self.pigment_name_to_id[p_name])
                    cluster['excluded_pigments'] = excluded_pigments
                else:
                    # Default for unknown colors
                    cluster['spectral_profile'] = {
                        'reflectance_peak': rgb,
                        'spectral_variance': 0.1
                    }
                    cluster['possible_pigments'] = []
                    cluster['excluded_pigments'] = []
            
            all_clusters.append(significant_clusters)
            
        return all_clusters
    
    def _find_optimal_clusters(self, pixels, max_attempt=5):
        """Find optimal number of clusters using silhouette analysis"""
        from sklearn.metrics import silhouette_score
        
        # Use a smaller sample for efficiency in silhouette calculation
        max_silhouette_samples = 1000
        if pixels.shape[0] > max_silhouette_samples:
            indices = np.random.choice(pixels.shape[0], max_silhouette_samples, replace=False)
            silhouette_sample = pixels[indices]
        else:
            silhouette_sample = pixels
            
        best_score = -1
        best_k = self.min_clusters
        
        # Simple color complexity heuristic based on color variance
        color_variance = np.var(pixels, axis=0).sum()
        
        # For very low variance, use fewer clusters
        if color_variance < 0.01:
            return max(2, self.min_clusters - 1)
        
        # For high variance, try more clusters but don't exceed max_clusters
        k_range = range(self.min_clusters, min(self.max_clusters + 1, 10))
        
        # Only try a few cluster values to avoid too much computation
        actual_k_values = np.linspace(min(k_range), max(k_range), 
                                      min(max_attempt, len(k_range)), 
                                      dtype=int).tolist()
        
        # Ensure no duplicates
        actual_k_values = sorted(list(set(actual_k_values)))
        
        for k in actual_k_values:
            try:
                # Only run for a few iterations to get an estimate
                kmeans = KMeans(n_clusters=k, random_state=42, max_iter=100, n_init=2)
                cluster_labels = kmeans.fit_predict(silhouette_sample)
                
                # Calculate silhouette score
                if len(np.unique(cluster_labels)) > 1:  # Only valid if >1 cluster found
                    score = silhouette_score(silhouette_sample, cluster_labels, random_state=42)
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception as e:
                print(f"Error in silhouette analysis for k={k}: {e}")
                continue
                
        return best_k
        
    def _rgb_to_color_name(self, rgb_values):
        """Map RGB values to predefined color categories using spectral profiles"""
        min_dist = float('inf')
        closest_color = 'other'
        
        for color_name, profile in self.spectral_profiles.items():
            peak = profile['reflectance_peak']
            variance = profile['spectral_variance']
            
            # Calculate distance with weighting by variance (more permissive for high-variance colors)
            weighted_dist = np.sqrt(sum([(a - b)**2 / max(variance, 0.01) 
                                       for a, b in zip(rgb_values, peak)]))
            
            if weighted_dist < min_dist:
                min_dist = weighted_dist
                closest_color = color_name
                
        return closest_color
    
    def _generate_color_histogram(self, image_tensor):
        """Generate color histogram from image tensor"""
        # Handle different shapes
        if len(image_tensor.shape) == 4:  # [B, C, H, W]
            batch_size, channels, height, width = image_tensor.shape
            # Reshape to [B, H*W, C]
            pixels = image_tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        elif len(image_tensor.shape) == 3:  # [C, H, W]
            channels, height, width = image_tensor.shape
            # Add batch dimension and reshape to [1, H*W, C]
            pixels = image_tensor.permute(1, 2, 0).reshape(1, -1, channels)
        elif len(image_tensor.shape) == 2:  # [B, C]
            return None  # Can't generate histogram from feature vector
        else:
            raise ValueError(f"Unsupported image tensor shape: {image_tensor.shape}")

        # Initialize histogram tensors
        batch_size = pixels.shape[0]
        histograms = []
        
        # For each image in batch
        for b in range(batch_size):
            # Get pixels for this image
            img_pixels = pixels[b]  # [H*W, C]
            
            # Initialize histogram (256 bins per channel)
            histogram = torch.zeros(3 * 256, device=img_pixels.device)
            
            # For each channel
            for c in range(3):
                channel_values = img_pixels[:, c]  # [H*W]
                
                # Scale to 0-255 and convert to integers for binning
                scaled_values = (channel_values * 255).long()
                scaled_values = torch.clamp(scaled_values, 0, 255)
                
                # Count occurrences of each value
                for i in range(256):
                    count = (scaled_values == i).sum().float()
                    histogram[c * 256 + i] = count
            
            # Normalize histogram
            if histogram.sum() > 0:
                histogram = histogram / histogram.sum()
                
            histograms.append(histogram)
            
        # Stack histograms into batch
        return torch.stack(histograms)
        
    def forward(self, features, image_tensor=None, color_histogram=None):
        """
        Predict physically realistic pigment distributions with adaptive clustering
        
        Args:
            features: Features from image encoder [batch_size, feat_dim]
            image_tensor: Optional raw image tensor for color analysis [batch_size, C, H, W]
            color_histogram: Optional pre-computed color histogram
            
        Returns:
            Dictionary with pigment probabilities and color clustering information
        """
        batch_size = features.shape[0]
        
        # Adapt features if needed
        if features.size(1) != self.swin_feat_dim:
            # Apply feature adaptation
            if features.size(1) == 3:  # Simple RGB features
                features = self.feature_adapter(features)
            else:
                # Use adaptive pooling for other dimensions
                features = F.adaptive_avg_pool1d(
                    features.unsqueeze(2), 
                    self.swin_feat_dim
                ).squeeze(2)
        
        # Generate color histogram if not provided
        if color_histogram is None and image_tensor is not None:
            color_histogram = self._generate_color_histogram(image_tensor)
            
        # If still no histogram, create a placeholder
        if color_histogram is None or color_histogram.size(1) != 3*256:
            print("Warning: Creating placeholder color histogram")
            color_histogram = torch.zeros((batch_size, 3*256), device=features.device)
            color_histogram = color_histogram + torch.rand_like(color_histogram) * 0.1
            
        # Extract color features
        color_feat = self.color_encoder(color_histogram)
        
        # Perform adaptive color clustering if image tensor is provided
        color_clusters = None
        if image_tensor is not None:
            color_clusters = self._adaptive_color_clustering(image_tensor)
        
        # Concatenate features
        joint_feat = torch.cat([features, color_feat], dim=1)
        
        # Predict raw pigment logits
        pigment_logits = self.joint_mlp(joint_feat)
        
        # Apply historical and cultural priors
        prior_weights = self.pigment_prior * self.style_modulation
        weighted_logits = pigment_logits * prior_weights
        
        # Convert to base probability distribution
        base_pigment_probs = F.softmax(weighted_logits, dim=1)
        
        # Apply constraints from color clustering if available
        constrained_probs = base_pigment_probs.clone()
        if color_clusters is not None:
            constrained_probs = self._apply_cluster_constraints(base_pigment_probs, color_clusters)
            
        return {
            'pigment_probs': constrained_probs,
            'base_probs': base_pigment_probs,
            'color_clusters': color_clusters
        }
        
    def _apply_cluster_constraints(self, base_probs, all_clusters):
        """
        Apply pigment constraints based on color clusters with realistic mixing
        
        This uses more advanced blending that considers:
        1. Percentage of each color in the image
        2. Plausible pigment combinations for each color
        3. Physical compatibility between pigments
        """
        batch_size = base_probs.size(0)
        constrained_probs = torch.zeros_like(base_probs)
        
        for b in range(batch_size):
            clusters = all_clusters[b]
            
            # Skip if no clusters
            if not clusters:
                constrained_probs[b] = base_probs[b]
                continue
                
            # Initialize weights for this image
            pigment_weights = torch.zeros(self.num_pigments, device=base_probs.device)
            
            # Process each cluster
            for cluster in clusters:
                # Get cluster information
                percentage = cluster['percentage']
                possible_pigments = cluster.get('possible_pigments', [])
                excluded_pigments = cluster.get('excluded_pigments', [])
                
                # Create mask for this cluster
                mask = torch.zeros(self.num_pigments, device=base_probs.device)
                
                if possible_pigments:
                    # Set 1.0 for possible pigments
                    mask[possible_pigments] = 1.0
                else:
                    # If no specific pigments defined, allow all
                    mask = torch.ones(self.num_pigments, device=base_probs.device)
                
                # Exclude specific pigments
                if excluded_pigments:
                    mask[excluded_pigments] = 0.0
                
                # Apply base probabilities with mask
                cluster_probs = base_probs[b] * mask
                
                # Renormalize
                if cluster_probs.sum() > 0:
                    cluster_probs = cluster_probs / cluster_probs.sum()
                    
                # Add to weights based on cluster percentage
                pigment_weights += cluster_probs * percentage
            
            # Ensure non-zero probabilities
            if pigment_weights.sum() > 0:
                # Normalize to get final probabilities
                constrained_probs[b] = pigment_weights / pigment_weights.sum()
            else:
                # Fallback to base probabilities
                constrained_probs[b] = base_probs[b]
                
        return constrained_probs
    
    def get_pigment_suggestions(self, image_tensor, n_suggestions=5):
        """
        Get pigment suggestions for an image with detailed explanations
        
        Args:
            image_tensor: Image tensor [1, C, H, W]
            n_suggestions: Number of pigment suggestions to return
            
        Returns:
            Dictionary with pigment suggestions and explanations
        """
        # Ensure batch dimension
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        # Extract simple features
        avg_color = F.adaptive_avg_pool2d(image_tensor, 1).squeeze(-1).squeeze(-1)
        
        # Generate color histogram
        color_histogram = self._generate_color_histogram(image_tensor)
        
        # Get color clusters
        color_clusters = self._adaptive_color_clustering(image_tensor)
        
        # Extract color features
        color_feat = self.color_encoder(color_histogram)
        
        # Create simple features for prediction
        features = self.feature_adapter(avg_color)
        
        # Concatenate features
        joint_feat = torch.cat([features, color_feat], dim=1)
        
        # Predict pigment probabilities
        pigment_logits = self.joint_mlp(joint_feat)
        
        # Apply priors
        prior_weights = self.pigment_prior * self.style_modulation
        weighted_logits = pigment_logits * prior_weights
        
        # Convert to probability distribution
        base_probs = F.softmax(weighted_logits, dim=1)
        
        # Apply color cluster constraints
        constrained_probs = self._apply_cluster_constraints(base_probs, color_clusters)
        
        # Get top pigment suggestions
        top_probs, top_indices = torch.topk(constrained_probs[0], min(n_suggestions, self.num_pigments))
        
        # Convert to list for output
        suggestions = []
        for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
            pigment_name = self.pigment_db.id_to_name[idx]
            
            # Find what colors this pigment is suitable for
            suitable_colors = []
            for color, profile in self.spectral_profiles.items():
                if pigment_name in profile['possible_pigments']:
                    suitable_colors.append(color)
                    
            explanation = f"Suggested for {', '.join(suitable_colors)} tones in the image"
            
            # Add chemical formula if available - using pigment_db.pigments directly
            # Instead of calling get_pigment_by_name which doesn't exist
            for pigment in self.pigment_db.pigments:
                if pigment.get('name') == pigment_name and 'formula' in pigment:
                    explanation += f". Chemical composition: {pigment['formula']}"
                    break
                
            suggestions.append({
                'rank': i + 1,
                'pigment_id': idx,
                'pigment_name': pigment_name,
                'probability': prob,
                'explanation': explanation
            })
            
        return {
            'suggestions': suggestions,
            'color_clusters': color_clusters[0]  # First image in batch
        }
    
    def analyze_image_pigments(self, image, features=None):
        """
        Main interface for PigmentModel to call - provides comprehensive pigment analysis
        
        Args:
            image: Input image tensor [B, C, H, W]
            features: Optional pre-extracted features
            
        Returns:
            Dictionary with pigment analysis including:
            - pigment_probabilities: Most likely pigments with probabilities
            - color_clusters: Detected color clusters in the image
            - pigment_suggestions: Specific pigment suggestions with explanations
            - layer_recommendations: Suggested pigment layering based on historical practices
        """
        # Ensure batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        # Extract features if not provided
        if features is None:
            # Extract average color as simple features
            features = F.adaptive_avg_pool2d(image, 1).squeeze(-1).squeeze(-1)
        
        # Get basic pigment data
        pigment_data = self.forward(features, image)
        
        # Get detailed pigment suggestions
        suggestions = self.get_pigment_suggestions(image)
        
        # Generate layer recommendations
        layer_recommendations = self._generate_layer_recommendations(pigment_data['color_clusters'])
        
        return {
            'pigment_probabilities': pigment_data['pigment_probs'],
            'color_clusters': pigment_data['color_clusters'],
            'pigment_suggestions': suggestions['suggestions'],
            'layer_recommendations': layer_recommendations
        }

    def _generate_layer_recommendations(self, color_clusters):
        """
        Generate historically accurate layer recommendations based on detected colors
        
        Args:
            color_clusters: Detected color clusters
            
        Returns:
            List of layer recommendations with pigment combinations
        """
        # Historical layering practices from Dunhuang murals
        historical_layering = {
            'ground_layer': ['kaolin', 'calcium_carbonate', 'gypsum'],
            'blue_layer': ['azurite', 'lapis_lazuli', 'synthetic_blue'],
            'green_layer': ['atacamite', 'malachite'],
            'red_layer': ['cinnabar', 'vermilion', 'hematite', 'red_ochre'],
            'yellow_layer': ['yellow_ochre', 'orpiment', 'arsenic_sulfide'],
            'highlights': ['lead_white', 'gold_leaf'],
            'outlines': ['carbon_black']
        }
        
        recommendations = []
        
        # Process all color clusters
        for batch in color_clusters:
            batch_recommendations = []
            
            # Create appropriate layer structure based on detected colors
            detected_colors = [cluster['color_name'] for cluster in batch]
            
            # Start with ground layer
            layer_structure = [{'name': 'ground_layer', 'pigments': historical_layering['ground_layer']}]
            
            # Add color layers based on detected colors
            if 'blue' in detected_colors:
                layer_structure.append({'name': 'blue_layer', 'pigments': historical_layering['blue_layer']})
                
            if 'green' in detected_colors:
                layer_structure.append({'name': 'green_layer', 'pigments': historical_layering['green_layer']})
                
            if 'red' in detected_colors:
                layer_structure.append({'name': 'red_layer', 'pigments': historical_layering['red_layer']})
                
            if 'yellow' in detected_colors:
                layer_structure.append({'name': 'yellow_layer', 'pigments': historical_layering['yellow_layer']})
                
            # Always add outlines and highlights as they were common in Dunhuang murals
            layer_structure.append({'name': 'outlines', 'pigments': historical_layering['outlines']})
            layer_structure.append({'name': 'highlights', 'pigments': historical_layering['highlights']})
            
            batch_recommendations.append(layer_structure)
            
            recommendations.append(batch_recommendations)
            
        return recommendations
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    
    # Initialize module
    color_to_pigment = ColorToPigmentMapper(num_pigments=35)
    color_to_pigment.eval()
    
    # Test 1: Basic functionality test
    def test_basic_functionality():
        print("\n=== Test 1: Basic Functionality ===")
        # Create a simple test image (red square)
        test_image = torch.zeros(1, 3, 64, 64)
        test_image[0, 0, :, :] = 0.9  # High red channel
        
        # Extract simple features
        avg_color = torch.tensor([[0.9, 0.0, 0.0]])  # Red color
        
        # Forward pass
        output = color_to_pigment(avg_color, test_image)
        
        print("Output keys:", list(output.keys()))
        print("Pigment probabilities shape:", output['pigment_probs'].shape)
        print("Color clusters:", len(output['color_clusters'][0]))
        
        # Verify shapes
        assert output['pigment_probs'].shape == (1, 35), "Pigment probability shape mismatch"
        assert len(output['color_clusters']) == 1, "Should have clusters for 1 image"
        
        # Verify probabilities sum to 1
        probs_sum = output['pigment_probs'].sum().item()
        print(f"Sum of probabilities: {probs_sum}")
        assert abs(probs_sum - 1.0) < 1e-5, "Probabilities should sum to 1"

    # Test 2: Color clustering test
    def test_color_clustering():
        print("\n=== Test 2: Color Clustering ===")
        # Create a test image with two distinct colors
        test_image = torch.zeros(1, 3, 64, 64)
        # Red in top half
        test_image[0, 0, :32, :] = 0.9
        # Blue in bottom half
        test_image[0, 2, 32:, :] = 0.9
        
        # Perform clustering
        clusters = color_to_pigment._adaptive_color_clustering(test_image)
        
        print(f"Number of clusters detected: {len(clusters[0])}")
        
        # Print cluster information
        for i, cluster in enumerate(clusters[0]):
            print(f"Cluster {i+1}:")
            print(f"  Color: {cluster['color_name']}")
            print(f"  RGB center: {cluster['center']}")
            print(f"  Percentage: {cluster['percentage']:.2f}")
            print(f"  Possible pigments: {len(cluster['possible_pigments'])}")
        
        # Verify we detect at least two clusters
        assert len(clusters[0]) >= 2, "Should detect at least 2 color clusters"
        
        # Verify cluster percentages sum to approximately 1
        total_percentage = sum(cluster['percentage'] for cluster in clusters[0])
        print(f"Total percentage: {total_percentage:.2f}")
        assert abs(total_percentage - 1.0) < 0.1, "Cluster percentages should sum to approximately 1"

    # Test 3: Color to name mapping test
    def test_color_name_mapping():
        print("\n=== Test 3: Color Name Mapping ===")
        # Test various RGB values
        test_colors = [
            (0.9, 0.1, 0.1),  # Red
            (0.1, 0.1, 0.9),  # Blue
            (0.1, 0.8, 0.1),  # Green
            (0.9, 0.8, 0.1),  # Yellow
            (0.1, 0.1, 0.1),  # Black
            (0.9, 0.9, 0.9),  # White
            (0.5, 0.5, 0.5),  # Gray
            (0.8, 0.4, 0.1),  # Orange
            (0.5, 0.1, 0.5),  # Purple
            (0.4, 0.3, 0.2)   # Brown
        ]
        
        for rgb in test_colors:
            color_name = color_to_pigment._rgb_to_color_name(rgb)
            print(f"RGB {rgb} -> {color_name}")
        
        # Verify specific mappings
        assert color_to_pigment._rgb_to_color_name((0.9, 0.1, 0.1)) == "red", "Should map to red"
        assert color_to_pigment._rgb_to_color_name((0.1, 0.1, 0.9)) == "blue", "Should map to blue"
        assert color_to_pigment._rgb_to_color_name((0.9, 0.9, 0.9)) == "white", "Should map to white"

    # Test 4: Pigment suggestion test
    def test_pigment_suggestions():
        print("\n=== Test 4: Pigment Suggestions ===")
        # Create a red test image
        test_image = torch.zeros(1, 3, 64, 64)
        test_image[0, 0, :, :] = 0.9  # High red channel
        
        # Get suggestions
        suggestions = color_to_pigment.get_pigment_suggestions(test_image, n_suggestions=3)
        
        print("Pigment suggestions:")
        for suggestion in suggestions['suggestions']:
            print(f"Rank {suggestion['rank']}: {suggestion['pigment_name']} (probability: {suggestion['probability']:.4f})")
            print(f"  Explanation: {suggestion['explanation']}")
        
        print("\nColor clusters in the image:")
        for i, cluster in enumerate(suggestions['color_clusters']):
            print(f"  {cluster['color_name']}: {cluster['percentage']:.2f}")
        
        # Verify structure
        assert 'suggestions' in suggestions, "Should return suggestions"
        assert 'color_clusters' in suggestions, "Should return color clusters"
        assert len(suggestions['suggestions']) <= 3, "Should return at most 3 suggestions"
        
        # For red image, verify we get a red pigment suggestion
        red_pigments = ['cinnabar', 'vermilion', 'hematite', 'red_ochre', 'lac']
        has_red_pigment = any(s['pigment_name'] in red_pigments for s in suggestions['suggestions'])
        assert has_red_pigment, "Should suggest a red pigment for a red image"

    # Test 5: Edge cases
    def test_edge_cases():
        print("\n=== Test 5: Edge Cases ===")
        
        # Test with an empty tensor
        try:
            output = color_to_pigment(torch.zeros(0, 3))
            print("Empty tensor handled without errors")
        except Exception as e:
            print("Empty tensor error:", str(e))
        
        # Test with very small image
        tiny_image = torch.rand(1, 3, 2, 2)
        tiny_features = torch.mean(tiny_image.reshape(1, 3, -1), dim=2)
        try:
            output = color_to_pigment(tiny_features, tiny_image)
            print("Tiny image handled successfully")
            print(f"Found {len(output['color_clusters'][0])} clusters in tiny image")
        except Exception as e:
            print("Tiny image error:", str(e))
        
        # Test with unusual dimensions
        unusual_features = torch.rand(1, 100)  # 100-dim features
        try:
            output = color_to_pigment(unusual_features)
            print("Unusual feature dimensions handled successfully")
            print(f"Output shape: {output['pigment_probs'].shape}")
        except Exception as e:
            print("Unusual dimensions error:", str(e))
        
        # Test with extreme values
        extreme_image = torch.ones(1, 3, 32, 32) * 100  # Very high values
        extreme_features = torch.ones(1, 3) * 100
        try:
            output = color_to_pigment(extreme_features, extreme_image)
            print("Extreme values handled successfully")
        except Exception as e:
            print("Extreme values error:", str(e))

    # Run all tests
    with torch.no_grad():
        test_basic_functionality()
        test_color_clustering()
        test_color_name_mapping()
        test_pigment_suggestions()
        test_edge_cases()
        
        print("\nAll tests completed!")