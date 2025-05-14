"""
Demo for Dunhuang Mural Physics-Based Pigment Analysis.

This script runs the Physics-Based Pigment Analysis pipeline with configuration
loaded from a JSON file, allowing for complete control over all parameters
including viewing angles, light directions, environmental conditions, etc.

Usage:
    python demo.py --config config.json
    python demo.py --config config.json --image path/to/image.png
    python demo.py --list-tests                 # Show available test presets
    python demo.py --run-test stability         # Run a specific test preset
"""
import os
import json
import time
import argparse
from datetime import datetime
import logging
import traceback
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms

from physics_refine import PhysicsRefinementModule


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("physics_analysis.log")
    ]
)
logger = logging.getLogger("physics_analysis")


def load_config(config_path, override_args=None):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        override_args: Dictionary of arguments to override from command line
        
    Returns:
        Dictionary with configuration parameters
    """
    default_config = {
        "output_dir": "physics_refinement_results",
        "use_gpu": True,
        "image": {
            "path": "path to the image...",
            "size": [256, 256],
            "batch_processing": False,
            "folder": None
        },
        "pigment_analysis": {
            "min_clusters": 8,
            "max_clusters": 20,
            "use_dbscan": True,
            "n_suggestions": 5
        },
        "stability_validation": {
            "binder_type": "animal_glue",
            "environmental_conditions": {
                "humidity": 0.5,
                "light_exposure": 0.5,
                "air_pollutants": 0.3,
                "temperature_fluctuation": 0.3,
                "temperature": 298
            }
        },
        "aging_simulation": {
            "years": 100,
            "environmental_conditions": {
                "humidity": 0.6,
                "light_exposure": 0.8,
                "air_pollutants": 0.4,
                "temperature_fluctuation": 0.5
            }
        },
        "optical_properties": {
            "viewing_angle": 0.4, 
            "light_direction": [0.0, 0.0, 1.0], 
            "particle_data": {
                "mean_size": 5.0,
                "std": 1.0
            },
            "fluorescence": False,
            "microstructure": {
                "porosity": 0.3,
                "rms_height": 0.8,
                "correlation_length": 5.0
            }
        },
        "layered_structure": {
            "layers": [
                {
                    "pigments": None,  # Will be filled based on analysis
                    "binder": "animal_glue",
                    "thickness": 20.0
                },
                {
                    "pigments": None,  # Will be filled based on analysis
                    "binder": "animal_glue",
                    "thickness": 20.0
                }
            ]
        },
        "historical_accuracy": {
            "period": "middle_tang",
            "region": "dunhuang"
        },
        "monte_carlo": {
            "n_samples": 50,
            "parameter_bounds": {
                "humidity": [0.3, 0.7],
                "light_exposure": [0.2, 0.8],
                "air_pollutants": [0.1, 0.5],
                "temperature": [288, 308]
            }
        },
        "visualization": {
            "show_plots": False,
            "save_images": True,
            "dpi": 150,
            "figsize": [18, 12],
            "generate_report": True
        },
        "tests_to_run": [
            "pigment_analysis",
            "stability",
            "aging",
            "optical",
            "layered",
            "historical",
            "rendering"
        ]
    }
    
    # Load config file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Merge with default config (recursive update)
                merge_configs(default_config, loaded_config)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.error(traceback.format_exc())
    
    # Apply command-line overrides
    if override_args:
        apply_overrides(default_config, override_args)
    
    return default_config


def merge_configs(base_config, override_config):
    """
    Recursively merge override_config into base_config.
    
    Args:
        base_config: Base configuration dictionary (modified in place)
        override_config: Override configuration dictionary
    """
    for key, value in override_config.items():
        if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
            # Recursive merge for nested dictionaries
            merge_configs(base_config[key], value)
        else:
            # Direct override for everything else
            base_config[key] = value


def apply_overrides(config, override_args):
    if 'image_path' in override_args and override_args['image_path']:
        config['image']['path'] = override_args['image_path']
    
    if 'output_dir' in override_args and override_args['output_dir']:
        config['output_dir'] = override_args['output_dir']
    
    if 'tests' in override_args and override_args['tests']:
        config['tests_to_run'] = override_args['tests'].split(',')



def create_output_directory(config):
    """
    Create output directory with timestamp for results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to created output directory
    """
    base_dir = config['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for test in config['tests_to_run']:
        os.makedirs(os.path.join(output_dir, test), exist_ok=True)

    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    return output_dir


def setup_device(config):
    """
    Set up the computation device based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch device
    """
    if config['use_gpu'] and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if config['use_gpu'] and not torch.cuda.is_available():
            logger.warning("GPU requested but not available. Using CPU instead.")
        else:
            logger.info("Using CPU for computations")
    
    return device


def load_image(config, device):
    """
    Load and preprocess the input image.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Preprocessed image tensor
    """
    image_path = config['image']['path']

    if not image_path or not os.path.exists(image_path):
        logger.warning(f"Image not found at {image_path}")
        logger.info("Creating synthetic test image...")

        image = Image.new('RGB', tuple(config['image']['size']), color=(255, 255, 255))
        pixels = image.load()

        width, height = config['image']['size']
        half_w, half_h = width // 2, height // 2

        for i in range(half_w):
            for j in range(half_h):
                pixels[i, j] = (220, 40, 40)

        for i in range(half_w, width):
            for j in range(half_h, height):
                pixels[i, j] = (40, 40, 200)

        for i in range(half_w):
            for j in range(half_h, height):
                pixels[i, j] = (220, 200, 40)

        for i in range(half_w, width):
            for j in range(half_h):
                pixels[i, j] = (40, 180, 80)

        synthetic_path = "test_image.png"
        image.save(synthetic_path)
        logger.info(f"Created synthetic test image at {synthetic_path}")
        image_path = synthetic_path
    else:
        logger.info(f"Loading image from {image_path}")
        image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(tuple(config['image']['size'])),
        transforms.ToTensor(),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)
    logger.info(f"Image tensor shape: {image_tensor.shape}")
    
    return image_tensor, image_path


def run_pigment_analysis(physics_module, image_tensor, config, output_dir):
    """
    Run pigment analysis on the input image.
    
    Args:
        physics_module: Initialized PhysicsRefinementModule
        image_tensor: Preprocessed image tensor
        config: Configuration dictionary
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("Running Pigment Analysis...")

    color_mapper_config = config.get('pigment_analysis', {})
    if hasattr(physics_module.color_mapper, 'min_clusters'):
        physics_module.color_mapper.min_clusters = color_mapper_config.get('min_clusters', 
                                                    physics_module.color_mapper.min_clusters)
    if hasattr(physics_module.color_mapper, 'max_clusters'):
        physics_module.color_mapper.max_clusters = color_mapper_config.get('max_clusters', 
                                                    physics_module.color_mapper.max_clusters)
    if hasattr(physics_module.color_mapper, 'use_dbscan'):
        physics_module.color_mapper.use_dbscan = color_mapper_config.get('use_dbscan', 
                                                physics_module.color_mapper.use_dbscan)

    start_time = time.time()
    analysis_result = physics_module.analyze_image(
        image_tensor, 
        n_suggestions=color_mapper_config.get('n_suggestions', 5)
    )
    analysis_time = time.time() - start_time
    logger.info(f"Analysis completed in {analysis_time:.2f} seconds")

    logger.info("\nTop pigment suggestions:")
    for i, suggestion in enumerate(analysis_result['pigment_suggestions'][:5], 1):
        logger.info(f"{i}. {suggestion['pigment_name']} (probability: {suggestion['probability']:.4f})")
        logger.info(f"   Explanation: {suggestion['explanation']}")

    logger.info("\nColor clusters:")
    for i, cluster in enumerate(analysis_result['color_clusters'][0][:3], 1):
        logger.info(f"{i}. Color: {cluster['color_name']}, Percentage: {cluster['percentage']*100:.2f}%")

    physics_module.visualize_results(
        analysis_result, 
        "pigment_analysis", 
        save_path=os.path.join(output_dir, "pigment_analysis", "analysis_results.png"),
        show=config['visualization']['show_plots']
    )
    logger.info(f"Visualization saved to {os.path.join(output_dir, 'pigment_analysis', 'analysis_results.png')}")

    physics_module.export_result_data(
        analysis_result, 
        format_type='json', 
        file_path=os.path.join(output_dir, "pigment_analysis", "analysis_data.json")
    )
    logger.info(f"Data exported to {os.path.join(output_dir, 'pigment_analysis', 'analysis_data.json')}")
    
    return analysis_result


def run_stability_validation(physics_module, analysis_result, config, output_dir):
    """
    Run pigment mixture stability validation.
    
    Args:
        physics_module: Initialized PhysicsRefinementModule
        analysis_result: Results from pigment analysis
        config: Configuration dictionary
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with stability validation results
    """
    logger.info("Running Stability Validation...")

    stability_config = config.get('stability_validation', {})
    binder_type = stability_config.get('binder_type', "animal_glue")
    env_conditions = stability_config.get('environmental_conditions', {})

    if 'pigment_suggestions' in analysis_result and len(analysis_result['pigment_suggestions']) >= 2:
        suggestion1 = analysis_result['pigment_suggestions'][0]
        suggestion2 = analysis_result['pigment_suggestions'][1]

        if 'pigment_id' in suggestion1 and 'pigment_id' in suggestion2:
            pigment1_id = suggestion1['pigment_id']
            pigment2_id = suggestion2['pigment_id']
            pigment1_name = suggestion1.get('pigment_name', f"Pigment {pigment1_id}")
            pigment2_name = suggestion2.get('pigment_name', f"Pigment {pigment2_id}")
        elif 'pigment_name' in suggestion1 and 'pigment_name' in suggestion2:
            pigment1_name = suggestion1['pigment_name']
            pigment2_name = suggestion2['pigment_name']
            pigment1_id = physics_module.pigment_db.name_to_id.get(pigment1_name)
            pigment2_id = physics_module.pigment_db.name_to_id.get(pigment2_name)
        else:
            logger.error("No pigment IDs or names found in analysis results. Cannot run stability validation.")
            return None

        if pigment1_id is not None and pigment2_id is not None:
            test_pigments = torch.tensor([[pigment1_id, pigment2_id]], device=physics_module.device)
            test_pigment_names = [pigment1_name, pigment2_name]
            logger.info(f"Testing mixture of detected pigments: {test_pigment_names}")
        else:
            logger.error("No pigment IDs found in analysis results. Cannot run stability validation.")
            return None
    else:
        logger.error("No pigments detected in image. Cannot run stability validation.")
        return None

    test_ratios = torch.tensor([[0.7, 0.3]], device=physics_module.device)

    start_time = time.time()
    stability_result = physics_module.validate_pigment_mixture(
        test_pigments,
        test_ratios,
        binder_type=binder_type,
        environmental_conditions=env_conditions
    )
    validation_time = time.time() - start_time
    logger.info(f"Validation completed in {validation_time:.2f} seconds")

    logger.info(f"\nStability score: {stability_result['stability_score']:.2f} "
          f"({stability_result['stability_rating']})")
    
    if stability_result['warnings']:
        logger.info("\nStability warnings:")
        for warning in stability_result['warnings']:
            logger.info(f"- {warning}")

    physics_module.visualize_results(
        stability_result, 
        "stability", 
        save_path=os.path.join(output_dir, "stability", "stability_analysis.png"),
        show=config['visualization']['show_plots']
    )
    logger.info(f"Visualization saved to {os.path.join(output_dir, 'stability', 'stability_analysis.png')}")

    physics_module.export_result_data(
        stability_result, 
        format_type='json', 
        file_path=os.path.join(output_dir, "stability", "stability_analysis.json")
    )
    logger.info(f"Data exported to {os.path.join(output_dir, 'stability', 'stability_analysis.json')}")
    
    return stability_result


def run_aging_simulation(physics_module, test_pigments, test_ratios, config, output_dir):
    """
    Run aging simulation.
    
    Args:
        physics_module: Initialized PhysicsRefinementModule
        test_pigments: Tensor of pigment indices
        test_ratios: Tensor of mixing ratios
        config: Configuration dictionary
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with aging simulation results
    """
    logger.info("Running Aging Simulation...")

    aging_config = config.get('aging_simulation', {})
    years = aging_config.get('years', 100)
    env_conditions = aging_config.get('environmental_conditions', {})

    start_time = time.time()
    aging_result = physics_module.simulate_aging(
        test_pigments,
        test_ratios,
        years=years,
        environmental_conditions=env_conditions
    )
    simulation_time = time.time() - start_time
    logger.info(f"Aging simulation completed in {simulation_time:.2f} seconds")

    logger.info(f"\nCondition after {years} years: {aging_result['condition_score']:.2f} "
          f"({aging_result['condition_rating']})")
    
    logger.info("\nAging effects:")
    for effect, value in aging_result['aging_effects'].items():
        logger.info(f"- {effect.capitalize()}: {value:.2f}")
    
    logger.info(f"\nPrimary effect: {aging_result['primary_effect']}")
    
    logger.info("\nRecommendations:")
    for recommendation in aging_result['recommendations']:
        logger.info(f"- {recommendation}")

    physics_module.visualize_results(
        aging_result, 
        "aging", 
        save_path=os.path.join(output_dir, "aging", "aging_simulation.png"),
        show=config['visualization']['show_plots']
    )
    logger.info(f"Visualization saved to {os.path.join(output_dir, 'aging', 'aging_simulation.png')}")

    physics_module.export_result_data(
        aging_result, 
        format_type='json', 
        file_path=os.path.join(output_dir, "aging", "aging_simulation.json")
    )
    logger.info(f"Data exported to {os.path.join(output_dir, 'aging', 'aging_simulation.json')}")
    
    return aging_result


def run_optical_properties(physics_module, test_pigments, test_ratios, config, output_dir):
    """
    Run optical properties calculation.
    
    Args:
        physics_module: Initialized PhysicsRefinementModule
        test_pigments: Tensor of pigment indices
        test_ratios: Tensor of mixing ratios
        config: Configuration dictionary
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with optical properties results
    """
    logger.info("Running Optical Properties Calculation...")

    optical_config = config.get('optical_properties', {})

    particle_data = None
    if 'particle_data' in optical_config:
        particle_data = {
            "mean": optical_config['particle_data'].get('mean_size', 5.0),
            "std": optical_config['particle_data'].get('std', 1.0)
        }

    microstructure_params = None
    if 'microstructure' in optical_config:
        microstructure_params = optical_config['microstructure']

    fluorescence_params = None
    if optical_config.get('fluorescence', False):
        device = physics_module.device
        fluorescence_params = {
            'emission_strength': torch.tensor([0.3], device=device),
            'excitation_peak': torch.tensor([0.5], device=device),
            'emission_peak': torch.tensor([0.6], device=device),
            'bandwidth': torch.tensor([0.4], device=device)
        }

    start_time = time.time()
    optical_result = physics_module.calculate_optical_properties(
        test_pigments,
        test_ratios,
        particle_data=particle_data,
        fluorescence_params=fluorescence_params,
        microstructure_params=microstructure_params
    )
    calculation_time = time.time() - start_time
    logger.info(f"Optical properties calculation completed in {calculation_time:.2f} seconds")

    logger.info("\nRGB color values:")
    rgb_values = optical_result['rgb'][0].cpu().detach().numpy()
    logger.info(f"R: {rgb_values[0]:.4f}, G: {rgb_values[1]:.4f}, B: {rgb_values[2]:.4f}")
    
    logger.info("\nBRDF parameters:")
    for param, value in optical_result['brdf_params'].items():
        if isinstance(value, torch.Tensor):
            if value.numel() > 1:
                continue
            else:
                value = value.item()
        logger.info(f"- {param}: {value:.4f}")

    physics_module.visualize_results(
        optical_result, 
        "optical", 
        save_path=os.path.join(output_dir, "optical", "optical_properties.png"),
        show=config['visualization']['show_plots']
    )
    logger.info(f"Visualization saved to {os.path.join(output_dir, 'optical', 'optical_properties.png')}")

    physics_module.export_result_data(
        optical_result, 
        format_type='json', 
        file_path=os.path.join(output_dir, "optical", "optical_properties.json")
    )
    logger.info(f"Data exported to {os.path.join(output_dir, 'optical', 'optical_properties.json')}")
    
    return optical_result


def run_layered_structure_analysis(physics_module, analysis_result, config, output_dir):
    """
    Run layered structure analysis.
    
    Args:
        physics_module: Initialized PhysicsRefinementModule
        analysis_result: Results from pigment analysis
        config: Configuration dictionary
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with layered structure analysis results
    """
    logger.info("Running Layered Structure Analysis...")

    if 'pigment_suggestions' in analysis_result and len(analysis_result['pigment_suggestions']) >= 2:
        pigment1_name = analysis_result['pigment_suggestions'][0]['pigment_name']
        pigment2_name = analysis_result['pigment_suggestions'][1]['pigment_name']
        pigment1_id = physics_module.pigment_db.name_to_id.get(pigment1_name)
        pigment2_id = physics_module.pigment_db.name_to_id.get(pigment2_name)

        if pigment1_id is not None and pigment2_id is not None:
            layer1 = torch.tensor([[pigment1_id]], device=physics_module.device)
            layer2 = torch.tensor([[pigment2_id]], device=physics_module.device)
            logger.info(f"Testing layered structure with {pigment1_name} over {pigment2_name}")
        else:
            logger.error("Could not find IDs for detected pigments. Cannot run layered structure analysis.")
            return None
    else:
        logger.error("Not enough pigments detected in image. Cannot run layered structure analysis.")
        return None

    layer_config = config.get('layered_structure', {})
    layer_binders = [layer['binder'] for layer in layer_config.get('layers', [{"binder": "animal_glue"}, {"binder": "animal_glue"}])]
    years = config['aging_simulation'].get('years', 50)
    env_conditions = config['aging_simulation'].get('environmental_conditions', {})

    start_time = time.time()
    layer_result = physics_module.analyze_layered_structure(
        [layer1, layer2],
        layer_binders=layer_binders,
        years=years,
        environmental_conditions=env_conditions
    )
    analysis_time = time.time() - start_time
    logger.info(f"Layered structure analysis completed in {analysis_time:.2f} seconds")

    logger.info(f"\nOverall stability score: {layer_result['overall_stability_score']:.2f}")
    logger.info(f"Is stable: {layer_result['is_stable']}")
    
    if 'interlayer_issues' in layer_result and layer_result['interlayer_issues']:
        logger.info("\nInterlayer issues detected:")
        for issue in layer_result['interlayer_issues']:
            if 'issues' in issue:
                for problem in issue['issues']:
                    logger.info(f"- {problem.get('description', 'Unknown issue')}")
                    logger.info(f"  Severity: {problem.get('severity', 'Unknown'):.2f}")

    physics_module.export_result_data(
        layer_result, 
        format_type='json', 
        file_path=os.path.join(output_dir, "layered", "layered_structure.json")
    )
    logger.info(f"Data exported to {os.path.join(output_dir, 'layered', 'layered_structure.json')}")
    
    return layer_result


def run_historical_accuracy(physics_module, test_pigments, test_ratios, config, output_dir):
    """
    Run historical accuracy analysis.
    
    Args:
        physics_module: Initialized PhysicsRefinementModule
        test_pigments: Tensor of pigment indices
        test_ratios: Tensor of mixing ratios
        config: Configuration dictionary
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with historical accuracy analysis results
    """
    logger.info("Running Historical Accuracy Analysis...")

    historical_config = config.get('historical_accuracy', {})
    historical_period = historical_config.get('period', "middle_tang")

    start_time = time.time()
    historical_result = physics_module.analyze_art_historical_accuracy(
        test_pigments,
        test_ratios,
        historical_period=historical_period
    )
    analysis_time = time.time() - start_time
    logger.info(f"Historical accuracy analysis completed in {analysis_time:.2f} seconds")
    logger.info(f"\nHistorical accuracy score: {historical_result['historical_accuracy_scores'][0]:.2f}")
    
    if 'period_appropriate_pigments' in historical_result:
        logger.info("\nPeriod-appropriate pigments:")
        for pid in historical_result['period_appropriate_pigments'][0]:
            name = physics_module.pigment_db.id_to_name.get(pid, f"Pigment {pid}")
            logger.info(f"- {name}")
    
    if 'anachronistic_pigments' in historical_result:
        logger.info("\nAnachronistic pigments:")
        for pid in historical_result['anachronistic_pigments'][0]:
            name = physics_module.pigment_db.id_to_name.get(pid, f"Pigment {pid}")
            logger.info(f"- {name}")
    
    if 'inappropriate_combinations' in historical_result:
        logger.info("\nHistorically inappropriate combinations:")
        for combo in historical_result['inappropriate_combinations']:
            logger.info(f"- {combo['combination'][0]} + {combo['combination'][1]}")
            logger.info(f"  Reason: {combo['reason']}")

    physics_module.visualize_results(
        historical_result, 
        "historical", 
        save_path=os.path.join(output_dir, "historical", "historical_accuracy.png"),
        show=config['visualization']['show_plots']
    )
    logger.info(f"Visualization saved to {os.path.join(output_dir, 'historical', 'historical_accuracy.png')}")

    physics_module.export_result_data(
        historical_result, 
        format_type='json', 
        file_path=os.path.join(output_dir, "historical", "historical_accuracy.json")
    )
    logger.info(f"Data exported to {os.path.join(output_dir, 'historical', 'historical_accuracy.json')}")
    
    return historical_result


def run_full_rendering(physics_module, test_pigments, test_ratios, config, output_dir):
    """
    Run full rendering simulation.
    
    Args:
        physics_module: Initialized PhysicsRefinementModule
        test_pigments: Tensor of pigment indices
        test_ratios: Tensor of mixing ratios
        config: Configuration dictionary
        output_dir: Output directory for saving results
        
    Returns:
        Dictionary with full rendering results
    """
    logger.info("Running Full Rendering Simulation...")

    optical_config = config.get('optical_properties', {})
    viewing_angle = optical_config.get('viewing_angle', 0.4)
    light_direction = optical_config.get('light_direction', [0.0, 0.0, 1.0])

    light_direction_tensor = torch.tensor(light_direction, device=physics_module.device)

    start_time = time.time()
    render_result = physics_module.simulate_full_rendering(
        test_pigments,
        test_ratios,
        age_years=config['aging_simulation'].get('years', 50),
        viewing_angle=viewing_angle,
        light_direction=light_direction_tensor,
        env_conditions=config['aging_simulation'].get('environmental_conditions', {})
    )
    rendering_time = time.time() - start_time
    logger.info(f"Full rendering simulation completed in {rendering_time:.2f} seconds")

    logger.info("\nRendered RGB values:")
    rgb_values = render_result['rgb'][0].cpu().detach().numpy()
    logger.info(f"R: {rgb_values[0]:.4f}, G: {rgb_values[1]:.4f}, B: {rgb_values[2]:.4f}")
    
    if 'view_dependent_rgb' in render_result:
        view_rgb = render_result['view_dependent_rgb'][0].cpu().detach().numpy()
        logger.info("\nView-dependent RGB values:")
        try:
            if view_rgb.shape == (3,):
                r, g, b = view_rgb[0], view_rgb[1], view_rgb[2]
                logger.info(f"R: {float(r):.4f}, G: {float(g):.4f}, B: {float(b):.4f}")
            elif len(view_rgb.shape) == 2 and view_rgb.shape[1] == 3:
                r, g, b = view_rgb[0][0], view_rgb[0][1], view_rgb[0][2]
                logger.info(f"R: {float(r):.4f}, G: {float(g):.4f}, B: {float(b):.4f}")
            elif len(view_rgb.shape) == 2 and view_rgb.shape[0] == 3:
                r, g, b = view_rgb[0][0], view_rgb[1][0], view_rgb[2][0]
                logger.info(f"R: {float(r):.4f}, G: {float(g):.4f}, B: {float(b):.4f}")
            else:
                logger.info(f"RGB values: {view_rgb}")
        except Exception as e:
            logger.error(f"Could not format RGB values: {e}")
            logger.info(f"Raw RGB data: {view_rgb}")

    physics_module.export_result_data(
        render_result, 
        format_type='json', 
        file_path=os.path.join(output_dir, "rendering", "full_rendering.json")
    )
    logger.info(f"Data exported to {os.path.join(output_dir, 'rendering', 'full_rendering.json')}")
    
    return render_result


def visualize_physics_refined_images(image_tensor, analysis_result, optical_result, render_result, 
                                    aging_result=None, save_path=None, show=False):
    """
    Create a comprehensive visualization of physics refined images
    
    Args:
        image_tensor: Original image tensor
        analysis_result: Pigment analysis result
        optical_result: Optical properties result
        render_result: Full rendering result
        aging_result: Optional aging simulation result
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    orig_img = image_tensor[0].permute(1, 2, 0).cpu().detach().numpy()

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
    
    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(orig_img)
    ax1.set_title("Original Image", fontsize=14)
    ax1.axis('off')
    
    # 2. Pigment Analysis Overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(orig_img)
    
    if analysis_result and 'color_clusters' in analysis_result:
        clusters = analysis_result['color_clusters'][0]
        for i, cluster in enumerate(clusters[:5]):
            color = cluster['center']
            percentage = cluster['percentage']
            ax2.add_patch(plt.Rectangle((10, 10 + i*25), 
                                    20, 20, color=color))
            color_name = cluster.get('color_name', 'Unknown')
            ax2.text(35, 25 + i*25, 
                    f"{color_name}: {percentage*100:.1f}%", 
                    fontsize=12, color='white', 
                    bbox=dict(facecolor='black', alpha=0.5))
    
    ax2.set_title("Pigment Analysis", fontsize=14)
    ax2.axis('off')
    
    # 3. Physics-Based Rendering
    ax3 = fig.add_subplot(gs[0, 2])

    if render_result and 'rgb' in render_result:
        if 'view_dependent_rgb' in render_result:
            render_rgb = render_result['view_dependent_rgb'][0].cpu().detach().numpy()
        else:
            render_rgb = render_result['rgb'][0].cpu().detach().numpy()

        rendered_img = orig_img.copy()
        
        for c in range(3): 
            if isinstance(render_rgb, torch.Tensor):
                render_rgb = render_rgb.cpu().detach().numpy()

            if render_rgb.ndim > 1:
                rgb_value = render_rgb[c][0] if render_rgb.shape[0] >= 3 else render_rgb[0][c]
            else:
                rgb_value = render_rgb[c]
            rendered_img[:, :, c] = rendered_img[:, :, c] * float(rgb_value)

        if 'brdf_params' in render_result:
            specular = render_result['brdf_params'].get('specular', 0.5)
            roughness = render_result['brdf_params'].get('roughness', 0.5)

            if isinstance(specular, torch.Tensor):
                specular = specular.item() if specular.numel() == 1 else specular.cpu().detach().numpy().mean()
            if isinstance(roughness, torch.Tensor):
                roughness = roughness.item() if roughness.numel() == 1 else roughness.cpu().detach().numpy().mean()

            h, w, _ = rendered_img.shape
            y, x = np.mgrid[0:h, 0:w]
            center_y, center_x = h//3, w//3

            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            max_dist = np.sqrt(h**2 + w**2)

            falloff = 20 + 100 * (1-roughness)
            spec_mask = np.exp(-dist**2 / falloff**2)

            spec_strength = float(specular) if hasattr(specular, 'item') else specular
            for c in range(3):
                rendered_img[:, :, c] = np.clip(
                    rendered_img[:, :, c] + spec_mask * spec_strength * 0.5, 
                    0, 1
                )
        
        ax3.imshow(rendered_img)
        ax3.set_title("Physics-Based Rendering", fontsize=14)
    else:
        ax3.imshow(orig_img)
        ax3.set_title("Physics-Based Rendering (No Data)", fontsize=14)
    
    ax3.axis('off')
    
    # 4. Optical Properties Visualization
    ax4 = fig.add_subplot(gs[1, 0])
    
    if optical_result and 'spectral' in optical_result:
        spectral_data = optical_result['spectral'][0].cpu().detach().numpy()
        wavelengths = np.linspace(400, 700, len(spectral_data))

        ax4_twin = ax4.twinx()
        ax4.plot(wavelengths, spectral_data, 'b-', linewidth=2)
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Reflectance', color='b')
        ax4.tick_params(axis='y', labelcolor='b')

        if 'brdf_params' in optical_result:
            brdf = optical_result['brdf_params']
            params = []
            labels = []
            
            for param in ['roughness', 'metallic', 'specular', 'anisotropy', 'subsurface']:
                if param in brdf:
                    value = brdf[param]
                    if torch.is_tensor(value):
                        if value.numel() > 1:
                            value = value.mean().item()
                        else:
                            value = value.item()
                    params.append(value)
                    labels.append(param.capitalize())
            
            if params:
                x = np.arange(len(params))
                ax4_twin.bar(x, params, alpha=0.5, color='r')
                ax4_twin.set_ylim(0, 1)
                ax4_twin.set_ylabel('BRDF Parameters', color='r')
                ax4_twin.tick_params(axis='y', labelcolor='r')
                ax4_twin.set_xticks(x)
                ax4_twin.set_xticklabels(labels, rotation=45)
    else:
        ax4.text(0.5, 0.5, "No optical data available", 
                ha='center', va='center', transform=ax4.transAxes)
    
    ax4.set_title("Optical Properties", fontsize=14)
    
    # 5. Aging Simulation
    ax5 = fig.add_subplot(gs[1, 1])
    
    if aging_result and 'color_change' in aging_result:
        aged_img = orig_img.copy()

        darkening = aging_result['aging_effects']['darkening']
        yellowing = aging_result['aging_effects']['yellowing']
        fading = aging_result['aging_effects']['fading']

        aged_img = aged_img * (1.0 - darkening * 0.7)
        aged_img[:, :, 2] *= (1.0 - yellowing * 0.7)  
        aged_img[:, :, 0] *= (1.0 + yellowing * 0.3)  
        aged_img[:, :, 1] *= (1.0 + yellowing * 0.5)

        for c in range(3):
            aged_img[:, :, c] = aged_img[:, :, c] * (1.0 - fading * 0.7) + fading * 0.7

        ax5.imshow(np.clip(aged_img, 0, 1))

        years = aging_result.get('years_simulated', 100)
        condition = aging_result.get('condition_rating', 'Unknown')
        
        ax5.text(10, 30, f"After {years} years", 
                fontsize=12, color='white', 
                bbox=dict(facecolor='black', alpha=0.7))
        ax5.text(10, 60, f"Condition: {condition}", 
                fontsize=12, color='white', 
                bbox=dict(facecolor='black', alpha=0.7))

        effects_text = ""
        for effect, value in aging_result['aging_effects'].items():
            effects_text += f"{effect.capitalize()}: {value:.2f}\n"
        
        ax5.text(10, 100, effects_text, 
                fontsize=10, color='white', 
                bbox=dict(facecolor='black', alpha=0.7))
        
        ax5.set_title(f"Aging Simulation ({years} years)", fontsize=14)
    else:
        ax5.imshow(orig_img)
        ax5.text(0.5, 0.5, "No aging simulation data", 
                ha='center', va='center', transform=ax5.transAxes,
                bbox=dict(facecolor='white', alpha=0.7))
        ax5.set_title("Aging Simulation", fontsize=14)
    
    ax5.axis('off')
    
    # 6. Spectral Visualization
    ax6 = fig.add_subplot(gs[1, 2])

    if 'spectral_data' in render_result or 'spectral' in optical_result:
        if 'spectral_data' in render_result:
            spectral = render_result['spectral_data'][0].cpu().detach().numpy()
        else:
            spectral = optical_result['spectral'][0].cpu().detach().numpy()

        wavelengths = np.linspace(400, 700, len(spectral))

        if 'rgb' in render_result:
            rgb = render_result['rgb'][0].cpu().detach().numpy()
        elif 'rgb' in optical_result:
            rgb = optical_result['rgb'][0].cpu().detach().numpy()
        else:
            rgb = [0.5, 0.5, 0.5] 

        h, w = 200, 400
        spec_img = np.zeros((h, w, 3))

        spec_img[:h//2, :, 0] = rgb[0]
        spec_img[:h//2, :, 1] = rgb[1]
        spec_img[:h//2, :, 2] = rgb[2]

        x = np.linspace(0, w-1, len(wavelengths))
        y = h - (spectral * h/2) - 1

        for i in range(1, len(x)):
            x1, y1 = int(x[i-1]), int(y[i-1])
            x2, y2 = int(x[i]), int(y[i])

            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                wl = wavelengths[i]
                r, g, b = wavelength_to_rgb(wl)
                
                if abs(x2-x1) > abs(y2-y1):
                    for xi in range(min(x1, x2), max(x1, x2)+1):
                        yi = y1 + (y2-y1) * (xi-x1) / (x2-x1) if x2 != x1 else y1
                        if 0 <= xi < w and 0 <= int(yi) < h:
                            spec_img[int(yi), xi] = [r, g, b]
                else:
                    for yi in range(min(y1, y2), max(y1, y2)+1):
                        xi = x1 + (x2-x1) * (yi-y1) / (y2-y1) if y2 != y1 else x1
                        if 0 <= int(xi) < w and 0 <= yi < h:
                            spec_img[yi, int(xi)] = [r, g, b]

        ax6.imshow(spec_img)

        for wl in [400, 500, 600, 700]:
            x_pos = int((wl - 400) / 300 * w)
            ax6.axvline(x_pos, color='white', alpha=0.5, linestyle='--')
            ax6.text(x_pos, h-20, f"{wl}nm", color='white', 
                    ha='center', bbox=dict(facecolor='black', alpha=0.7))
        
        ax6.set_title("Spectral Visualization", fontsize=14)
    else:
        ax6.text(0.5, 0.5, "No spectral data available", 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title("Spectral Visualization", fontsize=14)
    
    ax6.axis('off')

    plt.suptitle("Physics-Refined Image Analysis", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def wavelength_to_rgb(wavelength):
    """Convert wavelength to RGB color."""
    gamma = 0.8
    intensity_max = 1.0
    
    if wavelength < 380:
        return (0.0, 0.0, 0.0)
    elif wavelength < 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif wavelength < 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif wavelength < 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif wavelength < 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif wavelength < 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif wavelength <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        return (0.0, 0.0, 0.0)

    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 1.0
    
    r = pow(r * factor * intensity_max, gamma)
    g = pow(g * factor * intensity_max, gamma)
    b = pow(b * factor * intensity_max, gamma)
    
    return (r, g, b)


def generate_scientific_report(results, image_path, config, output_dir):
    """
    Generate a comprehensive scientific report in markdown format.
    
    Args:
        results: Dictionary containing all analysis results
        image_path: Path to the analyzed image
        config: Configuration used for the analysis
        output_dir: Output directory for saving the report
        
    Returns:
        Path to the generated report
    """
    report_path = os.path.join(output_dir, "scientific_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Scientific Analysis Report: Physics-Based Pigment Analysis\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Image:** {image_path}\n\n")

        f.write("## Analysis Parameters\n\n")
        f.write("The following parameters were used for this analysis:\n\n")
        f.write("```json\n")
        json.dump(config, f, indent=2)
        f.write("\n```\n\n")

        if 'analysis_result' in results:
            f.write("## 1. Pigment Analysis\n\n")
            f.write("### Identified Pigments\n\n")
            f.write("| Rank | Pigment | Probability | Explanation |\n")
            f.write("|------|---------|-------------|-------------|\n")
            for i, suggestion in enumerate(results['analysis_result']['pigment_suggestions'][:5], 1):
                prob = suggestion['probability']
                f.write(f"| {i} | {suggestion['pigment_name']} | {prob:.4f} | {suggestion['explanation']} |\n")
            
            f.write("\n### Color Clusters\n\n")
            f.write("| Color | Percentage |\n")
            f.write("|-------|------------|\n")
            for cluster in results['analysis_result']['color_clusters'][0][:5]:
                f.write(f"| {cluster['color_name']} | {cluster['percentage']*100:.2f}% |\n")
            
            f.write("\n![Pigment Analysis](./pigment_analysis/analysis_results.png)\n\n")
        
        if 'stability_result' in results:
            f.write("## 2. Stability Analysis\n\n")
            f.write(f"**Overall Stability Score:** {results['stability_result']['stability_score']:.2f} ")
            f.write(f"({results['stability_result']['stability_rating']})\n\n")
            
            if results['stability_result']['warnings']:
                f.write("### Stability Warnings\n\n")
                for warning in results['stability_result']['warnings']:
                    f.write(f"- {warning}\n")
            
            f.write("\n### Detailed Issues\n\n")
            f.write("| Issue Type | Impact | Description |\n")
            f.write("|------------|--------|-------------|\n")
            for issue in results['stability_result'].get('detailed_issues', []):
                issue_type = issue.get('type', 'Unknown')
                impact = issue.get('impact', 0)
                description = issue.get('description', 'No description available')
                f.write(f"| {issue_type} | {impact:.4f} | {description} |\n")
            
            f.write("\n![Stability Analysis](./stability/stability_analysis.png)\n\n")
        
        if 'aging_result' in results:
            f.write("## 3. Aging Simulation\n\n")
            f.write(f"**Years Simulated:** {results['aging_result']['years_simulated']}\n\n")
            f.write(f"**Condition After Aging:** {results['aging_result']['condition_score']:.2f} ")
            f.write(f"({results['aging_result']['condition_rating']})\n\n")
            
            f.write("### Aging Effects\n\n")
            f.write("| Effect | Magnitude |\n")
            f.write("|--------|------------|\n")
            for effect, value in results['aging_result']['aging_effects'].items():
                f.write(f"| {effect.capitalize()} | {value:.2f} |\n")
            
            f.write("\n**Primary Effect:** ")
            f.write(f"{results['aging_result']['primary_effect'].capitalize()}\n\n")
            
            f.write("### Color Change\n\n")
            f.write(f"**Delta E:** {results['aging_result']['color_change']['delta_e']:.2f}\n\n")
            
            f.write("### Recommendations\n\n")
            for recommendation in results['aging_result']['recommendations']:
                f.write(f"- {recommendation}\n")
            
            f.write("\n![Aging Simulation](./aging/aging_simulation.png)\n\n")
        
        if 'optical_result' in results:
            f.write("## 4. Optical Properties\n\n")
            
            if 'rgb' in results['optical_result']:
                rgb = results['optical_result']['rgb'][0].cpu().detach().numpy()
                f.write(f"**RGB Color:** R:{rgb[0]:.4f}, G:{rgb[1]:.4f}, B:{rgb[2]:.4f}\n\n")
            
            f.write("### BRDF Parameters\n\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            
            for param, value in results['optical_result']['brdf_params'].items():
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        value = value.item()
                    else:
                        continue
                f.write(f"| {param.capitalize()} | {value:.4f} |\n")
            
            f.write("\n![Optical Properties](./optical/optical_properties.png)\n\n")
        
        if 'layer_result' in results:
            f.write("## 5. Layered Structure Analysis\n\n")
            f.write(f"**Overall Stability Score:** {results['layer_result']['overall_stability_score']:.2f}\n\n")
            f.write(f"**Is Stable:** {results['layer_result']['is_stable']}\n\n")
            
            if 'interlayer_issues' in results['layer_result'] and results['layer_result']['interlayer_issues']:
                f.write("### Interlayer Issues\n\n")
                for issue in results['layer_result']['interlayer_issues']:
                    if 'issues' in issue:
                        for problem in issue['issues']:
                            f.write(f"- **{problem.get('type', 'Issue')}**: ")
                            f.write(f"{problem.get('description', 'No description')} ")
                            f.write(f"(Severity: {problem.get('severity', 'Unknown'):.2f})\n")
        
        if 'historical_result' in results:
            f.write("## 6. Historical Accuracy Analysis\n\n")
            f.write(f"**Historical Period:** {results['historical_result']['historical_period']}\n\n")
            f.write(f"**Historical Accuracy Score:** {results['historical_result']['historical_accuracy_scores'][0]:.2f}\n\n")
            
            if 'period_appropriate_pigments' in results['historical_result']:
                f.write("### Period-Appropriate Pigments\n\n")
                for pid in results['historical_result']['period_appropriate_pigments'][0]:
                    name = results['physics_module'].pigment_db.id_to_name.get(pid, f"Pigment {pid}")
                    f.write(f"- {name}\n")
            
            if 'anachronistic_pigments' in results['historical_result']:
                f.write("\n### Anachronistic Pigments\n\n")
                for pid in results['historical_result']['anachronistic_pigments'][0]:
                    name = results['physics_module'].pigment_db.id_to_name.get(pid, f"Pigment {pid}")
                    f.write(f"- {name}\n")
            
            f.write("\n![Historical Accuracy](./historical/historical_accuracy.png)\n\n")
        
        if 'render_result' in results:
            f.write("## 7. Physics-Based Rendering\n\n")
            
            if 'aged' in results['render_result']:
                f.write(f"**Aged Years:** {results['render_result'].get('age_years', 0)}\n\n")
            
            if 'brdf_params' in results['render_result']:
                f.write("### Rendering Parameters\n\n")
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")
                
                brdf = results['render_result']['brdf_params']
                params_to_show = ['roughness', 'metallic', 'specular', 'anisotropy', 'subsurface']
                
                for param in params_to_show:
                    if param in brdf:
                        value = brdf[param]
                        if isinstance(value, torch.Tensor):
                            if value.numel() == 1:
                                value = value.item()
                            else:
                                value = value.mean().item()
                        f.write(f"| {param.capitalize()} | {value:.4f} |\n")

        f.write("## Comprehensive Visualization\n\n")
        f.write("![Physics-Refined Image Analysis](./physics_refined_image.png)\n\n")

        f.write("## Conclusion\n\n")
        f.write("This report presents a comprehensive physics-based analysis of pigments and their properties. ")
        f.write("The analysis combines spectral data, chemical interactions, art historical knowledge, ")
        f.write("and physical rendering to provide insights into the materials and their behavior over time.\n\n")
        
        f.write("For further investigation, consider:\n\n")
        f.write("1. Testing different pigment combinations for improved stability\n")
        f.write("2. Exploring alternative viewing and lighting conditions\n")
        f.write("3. Comparing against historical reference samples\n")
    
    logger.info(f"Scientific report generated at {report_path}")
    return report_path


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Enhanced Physics-Based Pigment Analysis Demo")
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")
    parser.add_argument("--image", dest="image_path", type=str, help="Path to input image")
    parser.add_argument("--output", dest="output_dir", type=str, help="Output directory")
    parser.add_argument("--tests", type=str, help="Comma-separated list of tests to run")
    parser.add_argument("--list-tests", action="store_true", help="List available test presets")
    parser.add_argument("--run-test", type=str, help="Run a specific test preset")
    args = parser.parse_args()

    if args.list_tests:
        print("Available test presets:")
        print("  - all:        Run all tests")
        print("  - pigment:    Run pigment analysis only")
        print("  - stability:  Run stability validation only")
        print("  - aging:      Run aging simulation only")
        print("  - optical:    Run optical properties calculation only")
        print("  - layered:    Run layered structure analysis only")
        print("  - historical: Run historical accuracy analysis only")
        print("  - rendering:  Run full rendering simulation only")
        return

    if args.run_test:
        if args.run_test == "all":
            args.tests = "pigment_analysis,stability,aging,optical,layered,historical,rendering"
        elif args.run_test == "pigment":
            args.tests = "pigment_analysis"
        elif args.run_test == "stability":
            args.tests = "stability"
        elif args.run_test == "aging":
            args.tests = "aging"
        elif args.run_test == "optical":
            args.tests = "optical"
        elif args.run_test == "layered":
            args.tests = "layered"
        elif args.run_test == "historical":
            args.tests = "historical"
        elif args.run_test == "rendering":
            args.tests = "rendering"
        else:
            logger.error(f"Unknown test preset: {args.run_test}")
            return

    config = load_config(args.config, vars(args))

    output_dir = create_output_directory(config)
    logger.info(f"Results will be saved in {output_dir}")

    device = setup_device(config)

    logger.info("Initializing PhysicsRefinementModule...")
    start_time = time.time()
    physics_module = PhysicsRefinementModule(device=device)
    logger.info(f"Initialization complete in {time.time() - start_time:.2f} seconds")

    image_tensor, image_path = load_image(config, device)

    results = {
        'physics_module': physics_module
    }

    tests_to_run = config['tests_to_run']

    if "pigment_analysis" in tests_to_run:
        results['analysis_result'] = run_pigment_analysis(physics_module, image_tensor, config, output_dir)

    if 'analysis_result' in results and len(results['analysis_result']['pigment_suggestions']) >= 2:
        suggestion1 = results['analysis_result']['pigment_suggestions'][0]
        suggestion2 = results['analysis_result']['pigment_suggestions'][1]

        if 'pigment_id' in suggestion1 and 'pigment_id' in suggestion2:
            pigment1_id = suggestion1['pigment_id']
            pigment2_id = suggestion2['pigment_id']
        elif 'pigment_name' in suggestion1 and 'pigment_name' in suggestion2:
            pigment1_name = suggestion1['pigment_name']
            pigment2_name = suggestion2['pigment_name']
            pigment1_id = physics_module.pigment_db.name_to_id.get(pigment1_name)
            pigment2_id = physics_module.pigment_db.name_to_id.get(pigment2_name)
        else:
            pigment1_id = None
            pigment2_id = None

        if pigment1_id is not None and pigment2_id is not None:
            test_pigments = torch.tensor([[pigment1_id, pigment2_id]], device=device)
            test_ratios = torch.tensor([[0.7, 0.3]], device=device)
        else:
            test_pigments = torch.tensor([[10, 17]], device=device) 
            test_ratios = torch.tensor([[0.5, 0.5]], device=device)
    else:
        test_pigments = torch.tensor([[10, 17]], device=device) 
        test_ratios = torch.tensor([[0.5, 0.5]], device=device)

    if "stability" in tests_to_run:
        results['stability_result'] = run_stability_validation(
            physics_module, 
            results.get('analysis_result', {}), 
            config, 
            output_dir
        )

    if "aging" in tests_to_run:
        results['aging_result'] = run_aging_simulation(
            physics_module, 
            test_pigments, 
            test_ratios, 
            config, 
            output_dir
        )

    if "optical" in tests_to_run:
        results['optical_result'] = run_optical_properties(
            physics_module, 
            test_pigments, 
            test_ratios, 
            config, 
            output_dir
        )

    if "layered" in tests_to_run:
        results['layer_result'] = run_layered_structure_analysis(
            physics_module, 
            results.get('analysis_result', {}),
            config, 
            output_dir
        )

    if "historical" in tests_to_run:
        results['historical_result'] = run_historical_accuracy(
            physics_module, 
            test_pigments, 
            test_ratios, 
            config, 
            output_dir
        )

    if "rendering" in tests_to_run:
        results['render_result'] = run_full_rendering(
            physics_module, 
            test_pigments, 
            test_ratios, 
            config, 
            output_dir
        )

    if 'analysis_result' in results and ('render_result' in results or 'optical_result' in results):
        logger.info("Generating comprehensive physics-refined visualization...")
        visualize_physics_refined_images(
            image_tensor=image_tensor,
            analysis_result=results.get('analysis_result', None),
            optical_result=results.get('optical_result', None),
            render_result=results.get('render_result', None),
            aging_result=results.get('aging_result', None),
            save_path=os.path.join(output_dir, "physics_refined_image.png"),
            show=config['visualization']['show_plots']
        )
        logger.info(f"Comprehensive visualization saved to {os.path.join(output_dir, 'physics_refined_image.png')}")

    if config['visualization'].get('generate_report', True):
        report_path = generate_scientific_report(results, image_path, config, output_dir)
    
    logger.info("\n" + "="*50)
    logger.info("All Physics Refinement Tests Completed!")
    logger.info("="*50)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
