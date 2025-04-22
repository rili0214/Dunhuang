import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchvision import transforms
import argparse
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from physics.pigment_model import PigmentModel, DunhuangPigmentDB


class MuralPigmentTester:
    def __init__(self, model_path=None, device='cpu'):
        print("Initializing Mural Pigment Tester...")
        
        self.device = device
        
        # Initialize the integrated pigment model
        self.pigment_model = PigmentModel(
            num_pigments=16,
            feature_dim=768,  # Match the model's feature dimension
            hidden_dim=256,
            k=3,  # Top-3 pigments per pixel
            in_chans=3
        ).to(device)
        
        # Initialize pigment database for reference
        self.pigment_db = DunhuangPigmentDB()
        
        # Load model weights if provided, If you use SWIN Transformer, make sure it's v2 base model.
        if model_path and os.path.exists(model_path):
            print(f"Loading model weights from {model_path}")
            self.pigment_model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        print("Initialization complete")
    
    def load_image(self, image_path):
        """Load and preprocess a mural image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"Loading image from {image_path}")
        
        # Load and convert to RGB if needed
        image = Image.open(image_path).convert('RGB')
        
        # Keep original for display
        self.original_image = image
        
        # Transform for the model
        image_tensor = self.transform(image).to(self.device)
        self.image_tensor = image_tensor
        
        print(f"Image loaded with shape: {tuple(image_tensor.shape)}")
        return image_tensor
    
    def process_image(self):
        """Process the loaded image with the PigmentModel"""
        if not hasattr(self, 'image_tensor'):
            raise ValueError("No image loaded. Call load_image() first.")
        
        print("Processing image with PigmentModel...")
        
        # Add batch dimension if necessary
        image_tensor = self.image_tensor
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Use the pigment model's methods directly
        with torch.no_grad():
            # Extract features using the model's feature extraction method
            features = self.pigment_model._extract_pigment_features(image_tensor)
            
            # Generate color histogram
            if hasattr(self.pigment_model, 'color_histogram_generator'):
                color_histogram = self.pigment_model.color_histogram_generator(image_tensor)
            else:
                # Fallback: Create simplified histogram, which should never be used
                hist_r = torch.histc(image_tensor[0, 0], bins=256, min=0, max=1)
                hist_g = torch.histc(image_tensor[0, 1], bins=256, min=0, max=1)
                hist_b = torch.histc(image_tensor[0, 2], bins=256, min=0, max=1)
                color_histogram = torch.cat([hist_r, hist_g, hist_b]).unsqueeze(0)
            
            # Get dominant colors using the model's method
            dominant_color = self.pigment_model.detect_dominant_colors(image_tensor)
                
            # Process with the model
            pigment_info = self.pigment_model(features, color_histogram, dominant_color)
            
            # Create pigment layers using model's methods
            pigment_layers = self.pigment_model._decompose_to_pigment_layers(image_tensor, pigment_info)
            
            # Create results dict
            self.results = {
                'restored_image': image_tensor,
                'initial_image': image_tensor,
                'pigment_info': pigment_info,
                'pigment_layers': pigment_layers
            }
        
        print("Image processing complete")
        return self.results
    
    def visualize_original_and_refined(self):
        """Visualize the original image and the pigment-refined version"""
        if not hasattr(self, 'results'):
            raise ValueError("No processing results. Call process_image() first.")
        
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.original_image)
        plt.title("Original Mural Image")
        plt.axis('off')
        
        # Refined image
        plt.subplot(1, 2, 2)
        refined_image = self.results['restored_image'][0].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(np.clip(refined_image, 0, 1))
        plt.title("Pigment Model Analysis")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("mural_refined.png", dpi=200)
        plt.show()
    
    def visualize_pigment_analysis(self):
        """Visualize detailed pigment analysis"""
        if not hasattr(self, 'results'):
            raise ValueError("No processing results. Call process_image() first.")
        
        # Extract pigment information
        pigment_info = self.results['pigment_info']
        top_pigments = pigment_info['top_pigments'][0].cpu().numpy()
        selected_ratios = pigment_info['selected_ratios'][0].cpu().numpy()
        
        if 'age_factor' in pigment_info:
            age_factor = pigment_info['age_factor'][0].cpu().item()
        else:
            age_factor = 0.5  # Default if not provided
            
        if 'reflectance' in pigment_info:
            reflectance = pigment_info['reflectance'][0].cpu().numpy()
        else:
            reflectance = np.array([0.5, 0.5, 0.5])  # Default if not provided
            
        if 'roughness' in pigment_info:
            if hasattr(pigment_info['roughness'][0], 'cpu'):
                roughness = pigment_info['roughness'][0].cpu().item()
            else:
                roughness = pigment_info['roughness'][0].item()
        else:
            roughness = 0.4  # Default if not provided
        
        # Get names for pigments
        pigment_names = [self.pigment_db.id_to_name.get(p, f"Unknown-{p}") for p in top_pigments]
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # 1. Original image
        plt.subplot(2, 3, 1)
        plt.imshow(self.original_image)
        plt.title("Original Mural Image")
        plt.axis('off')
        
        # 2. Refined image
        plt.subplot(2, 3, 2)
        refined_image = self.results['restored_image'][0].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(np.clip(refined_image, 0, 1))
        plt.title("Pigment Analysis")
        plt.axis('off')
        
        # 3. Pigment distribution
        plt.subplot(2, 3, 3)
        colors = ['#E63946', '#457B9D', '#A8DADC']  # Red, blue, greenish
        plt.bar(range(len(top_pigments)), selected_ratios, color=colors)
        plt.title("Identified Pigments and Mixing Ratios")
        plt.xticks(range(len(top_pigments)), pigment_names, rotation=45, ha='right')
        plt.ylabel("Ratio")
        
        # 4. Pigment reflectance
        plt.subplot(2, 3, 4)
        plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color=reflectance)
        plt.title(f"Mixed Pigment Reflectance: {np.round(reflectance, 2)}")
        plt.axis('off')
        
        # 5. Age factor
        plt.subplot(2, 3, 5)
        # Create age gradient visualization
        gradient = np.linspace(0, 1, 100)
        gradient = np.vstack((gradient, gradient))
        plt.imshow(gradient, cmap='YlOrBr', aspect='auto')
        plt.title(f"Estimated Age Factor: {age_factor:.2f}")
        plt.axis('off')
        
        # 6. Roughness
        plt.subplot(2, 3, 6)
        # Create roughness texture visualization
        h, w = 100, 100
        x = np.linspace(0, 20*np.pi, h)
        y = np.linspace(0, 20*np.pi, w)
        X, Y = np.meshgrid(x, y)
        roughness_vis = np.sin(X) * np.sin(Y) * roughness
        plt.imshow(roughness_vis, cmap='gray')
        plt.title(f"Surface Roughness: {roughness:.2f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("pigment_analysis.png", dpi=200)
        plt.show()
        
        # Print summary
        print("\nPigment Analysis Summary:")
        print(f"Estimated Age Factor: {age_factor:.2f}")
        print(f"Surface Roughness: {roughness:.2f}")
        print("\nIdentified Pigments:")
        for name, ratio in zip(pigment_names, selected_ratios):
            print(f" - {name}: {ratio:.2f}")
            
        # Print stability info if available
        if 'stability_info' in pigment_info:
            stability_info = pigment_info['stability_info'][0]
            print("\nPigment Stability Analysis:")
            if 'stability_score' in stability_info:
                print(f"Stability Score: {stability_info['stability_score']:.2f}")
            if 'warnings' in stability_info and stability_info['warnings']:
                print("Stability Warnings:")
                for warning in stability_info['warnings']:
                    print(f" - {warning}")
    
    def visualize_layer_decomposition(self):
        """Visualize the decomposition into pigment layers"""
        if not hasattr(self, 'results'):
            raise ValueError("No processing results. Call process_image() first.")
        
        # Get pigment layers
        pigment_layers = self.results['pigment_layers']
        
        # Get pigment information
        pigment_info = self.results['pigment_info']
        top_pigments = pigment_info['top_pigments'][0].cpu().numpy()
        selected_ratios = pigment_info['selected_ratios'][0].cpu().numpy()
        
        # Get names for pigments
        pigment_names = [self.pigment_db.id_to_name.get(p, f"Unknown-{p}") for p in top_pigments]
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(2, len(pigment_layers) + 1, 1)
        plt.imshow(self.original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Refined image
        plt.subplot(2, len(pigment_layers) + 1, len(pigment_layers) + 1)
        refined_image = self.results['restored_image'][0].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(np.clip(refined_image, 0, 1))
        plt.title("Analyzed Image")
        plt.axis('off')
        
        # Individual pigment layers
        for i, layer in enumerate(pigment_layers):
            # Convert layer to numpy
            layer_np = layer[0].permute(1, 2, 0).detach().cpu().numpy()
            
            # Display layer
            plt.subplot(2, len(pigment_layers) + 1, i + 2)
            plt.imshow(np.clip(layer_np, 0, 1))
            plt.title(f"Layer {i+1}: {pigment_names[i]}\nRatio: {selected_ratios[i]:.2f}")
            plt.axis('off')
            
            # Get raw pigment color
            if top_pigments[i] < len(self.pigment_db.pigments):
                pigment_color = self.pigment_db.pigments[top_pigments[i]]["reflectance"]
            else:
                pigment_color = [0.5, 0.5, 0.5]  # Default
            
            # Display pure pigment color
            plt.subplot(2, len(pigment_layers) + 1, len(pigment_layers) + 1 + i + 1)
            plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color=pigment_color)
            plt.title(f"Pure {pigment_names[i]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("layer_decomposition.png", dpi=200)
        plt.show()
    
    def visualize_aging_simulation(self):
        """Simulate and visualize the aging process of the mural"""
        if not hasattr(self, 'results'):
            raise ValueError("No processing results. Call process_image() first.")
        
        # Get current age factor
        if 'age_factor' in self.results['pigment_info']:
            current_age = self.results['pigment_info']['age_factor'][0].cpu().item()
        else:
            current_age = 0.5  # Default if not provided
        
        # Create variants with different aging
        age_variants = [0.0, current_age * 0.5, current_age, current_age * 1.5, min(current_age * 2.0, 0.95)]
        age_labels = ["New", "Less Aged", "Current", "More Aged", "Much More Aged"]
        
        # Manually apply simplified aging effects to the image
        restored_rgb = self.image_tensor.permute(1, 2, 0).detach().cpu().numpy()
        
        # Create visualization
        plt.figure(figsize=(15, 6))
        
        for i, (age, label) in enumerate(zip(age_variants, age_labels)):
            # Apply simplified aging effects
            age_diff = age - current_age
            
            # Create aged variant
            aged_variant = restored_rgb.copy()
            
            if age_diff != 0:
                # Yellowing effect (increases red and green, decreases blue)
                aging_mask = np.array([0.3, 0.18, -0.12]) * age_diff
                
                # Apply the aging mask to each channel
                for c in range(3):
                    aged_variant[:, :, c] = aged_variant[:, :, c] * (1.0 + aging_mask[c])
                
                # Ensure valid range
                aged_variant = np.clip(aged_variant, 0, 1)
            
            # Display variant
            plt.subplot(1, len(age_variants), i+1)
            plt.imshow(aged_variant)
            plt.title(f"{label}\nAge Factor: {age:.2f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("aging_simulation.png", dpi=200)
        plt.show()
    
    def visualize_chemical_analysis(self):
        """Visualize the chemical analysis of pigments based on the model's output"""
        if not hasattr(self, 'results'):
            raise ValueError("No processing results. Call process_image() first.")
        
        # Get pigment information
        pigment_info = self.results['pigment_info']
        top_pigments = pigment_info['top_pigments'][0].cpu().numpy()
        
        # Get chemical formulas and compositions
        formulas = []
        elements = defaultdict(float)
        
        for p_id in top_pigments:
            if p_id < len(self.pigment_db.pigments):
                pigment = self.pigment_db.pigments[p_id]
                formula = pigment["formula"]
                formulas.append(formula)
                
                # Parse chemical composition
                composition = self.pigment_db._parse_chemical_formula(formula)
                for element, count in composition.items():
                    elements[element] += count
            else:
                formulas.append("Unknown")
        
        # Create element distribution visualization
        plt.figure(figsize=(12, 8))
        
        # Pigment formulas
        plt.subplot(2, 1, 1)
        pigment_names = [self.pigment_db.id_to_name.get(p, f"Unknown-{p}") for p in top_pigments]
        y_pos = np.arange(len(pigment_names))
        
        plt.barh(y_pos, [1] * len(pigment_names), color=['#AEC7E8', '#FFBB78', '#98DF8A'])
        plt.yticks(y_pos, [f"{name} ({formula})" for name, formula in zip(pigment_names, formulas)])
        plt.title("Pigment Chemical Formulas")
        plt.xlabel("Presence")
        
        # Element distribution
        plt.subplot(2, 1, 2)
        elements_list = list(elements.keys())
        counts = [elements[e] for e in elements_list]
        
        # Sort by count
        sorted_indices = np.argsort(counts)
        sorted_elements = [elements_list[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        # Element colors based on periodic table groups
        element_colors = {
            'H': '#AEC7E8',     # Light blue
            'O': '#FFBB78',     # Light orange
            'C': '#98DF8A',     # Light green
            'N': '#FF9896',     # Light red
            'S': '#C5B0D5',     # Light purple
            'Ca': '#C49C94',    # Light brown
            'Fe': '#F7B6D2',    # Light pink
            'Cu': '#DBDB8D',    # Light yellow-green
            'Pb': '#9EDAE5',    # Light cyan
            'Hg': '#BCBD22',    # Dark yellow-green
            'As': '#17BECF'     # Dark cyan
        }
        
        # Default color for other elements
        default_color = '#7F7F7F'  # Gray
        
        colors = [element_colors.get(e, default_color) for e in sorted_elements]
        
        plt.barh(np.arange(len(sorted_elements)), sorted_counts, color=colors)
        plt.yticks(np.arange(len(sorted_elements)), sorted_elements)
        plt.title("Element Distribution in Identified Pigments")
        plt.xlabel("Count")
        
        plt.tight_layout()
        plt.savefig("chemical_analysis.png", dpi=200)
        plt.show()
    
    def visualize_spectral_reflectance(self):
        """Visualize the spectral reflectance properties of identified pigments"""
        if not hasattr(self, 'results'):
            raise ValueError("No processing results. Call process_image() first.")
        
        # Get pigment information
        pigment_info = self.results['pigment_info']
        top_pigments = pigment_info['top_pigments'][0].cpu().numpy()
        selected_ratios = pigment_info['selected_ratios'][0].cpu().numpy()
        
        # Get pigment names
        pigment_names = [self.pigment_db.id_to_name.get(p, f"Unknown-{p}") for p in top_pigments]
        
        # Create wavelength range (approximated for RGB visualization)
        wavelengths = np.linspace(400, 700, 100)  # 400-700nm visible spectrum
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Plot individual pigment reflectance curves
        plt.subplot(1, 2, 1)
        
        for i, pigment_id in enumerate(top_pigments):
            if pigment_id < len(self.pigment_db.pigments):
                pigment = self.pigment_db.pigments[pigment_id]
                reflectance = pigment["reflectance"]
                
                # Approximate reflectance curve from RGB values
                # Map RGB to approximate spectral reflectance (simplified)
                r, g, b = reflectance
                curve = np.zeros_like(wavelengths)
                
                # Red component (peaks around 650nm)
                curve += r * np.exp(-((wavelengths - 650)**2) / (2 * 50**2))
                
                # Green component (peaks around 550nm)
                curve += g * np.exp(-((wavelengths - 550)**2) / (2 * 50**2))
                
                # Blue component (peaks around 450nm)
                curve += b * np.exp(-((wavelengths - 450)**2) / (2 * 50**2))
                
                # Normalize
                curve = curve / np.max(curve) if np.max(curve) > 0 else curve
                
                # Plot with color based on the pigment's RGB value
                plt.plot(wavelengths, curve, label=f"{pigment_names[i]} ({selected_ratios[i]:.2f})", 
                         color=reflectance, linewidth=2 + selected_ratios[i] * 3)
        
        plt.title("Approximate Spectral Reflectance of Identified Pigments")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.xlim(400, 700)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot mixed reflectance (combined effect)
        plt.subplot(1, 2, 2)
        
        # Get the mixed reflectance
        if 'reflectance' in pigment_info:
            mixed_reflectance = pigment_info['reflectance'][0].cpu().numpy()
        else:
            # Compute weighted average if not provided
            mixed_reflectance = np.zeros(3)
            for i, pigment_id in enumerate(top_pigments):
                if pigment_id < len(self.pigment_db.pigments):
                    mixed_reflectance += np.array(self.pigment_db.pigments[pigment_id]["reflectance"]) * selected_ratios[i]
        
        # Approximate mixed spectral curve from RGB
        r, g, b = mixed_reflectance
        mixed_curve = np.zeros_like(wavelengths)
        
        # Red component (peaks around 650nm)
        mixed_curve += r * np.exp(-((wavelengths - 650)**2) / (2 * 50**2))
        
        # Green component (peaks around 550nm)
        mixed_curve += g * np.exp(-((wavelengths - 550)**2) / (2 * 50**2))
        
        # Blue component (peaks around 450nm)
        mixed_curve += b * np.exp(-((wavelengths - 450)**2) / (2 * 50**2))
        
        # Normalize
        mixed_curve = mixed_curve / np.max(mixed_curve) if np.max(mixed_curve) > 0 else mixed_curve
        
        # Plot mixed reflectance
        plt.plot(wavelengths, mixed_curve, color=mixed_reflectance, linewidth=4)
        
        # Add color bar to show the mixed color
        color_bar = np.ones((10, len(wavelengths), 3))
        color_bar[:, :, 0] = mixed_reflectance[0]
        color_bar[:, :, 1] = mixed_reflectance[1]
        color_bar[:, :, 2] = mixed_reflectance[2]
        
        plt.imshow(color_bar, extent=[400, 700, -0.1, 0], aspect='auto')
        
        plt.title("Mixed Pigment Reflectance")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.xlim(400, 700)
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("spectral_reflectance.png", dpi=200)
        plt.show()
    
    def visualize_all(self):
        """Run all visualizations"""
        self.visualize_original_and_refined()
        self.visualize_pigment_analysis()
        self.visualize_layer_decomposition()
        self.visualize_aging_simulation()
        self.visualize_chemical_analysis()
        self.visualize_spectral_reflectance()  # Added new visualization
        
        print("\nAll visualizations completed and saved.")
        print("Files saved:")
        print(" - mural_refined.png")
        print(" - pigment_analysis.png")
        print(" - layer_decomposition.png")
        print(" - aging_simulation.png")
        print(" - chemical_analysis.png")
        print(" - spectral_reflectance.png")


def main():
    parser = argparse.ArgumentParser(description='Test the Dunhuang Pigment Model on mural images')
    parser.add_argument('--image_path', type=str, 
                        default='', 
                        help='Path to the input mural image')
    parser.add_argument('--model_path', type=str, help='Path to model weights (optional)', default=None)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to run the model on')
    args = parser.parse_args()
    
    # Check if CUDA is available when requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Create tester
    tester = MuralPigmentTester(model_path=args.model_path, device=args.device)
    
    # Run the test pipeline
    print(f"Testing image: {args.image_path}")
    tester.load_image(args.image_path)
    tester.process_image()
    tester.visualize_all()

if __name__ == "__main__":
    main()