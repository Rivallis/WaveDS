"""
WaveDS Dataset Usage Examples

This module demonstrates how to work with the WaveDS dataset for domain shift research
in ultrasonic wavefield imaging.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class WaveDSLoader:
    """
    Data loader for the WaveDS ultrasonic wavefield domain shift dataset.
    
    This class provides utilities to load measurement data, metadata, and 
    organize domain splits for machine learning experiments.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the WaveDS data loader.
        
        Args:
            dataset_path: Path to the WaveDS dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.data_path = self.dataset_path / "data"
        self.metadata_path = self.dataset_path / "metadata"
        
        # Load metadata tables
        self._load_metadata()
    
    def _load_metadata(self):
        """Load all metadata tables into memory."""
        try:
            self.specimens = pd.read_csv(self.metadata_path / "specimen_specifications.csv")
            self.defects = pd.read_csv(self.metadata_path / "defect_parameters.csv")
            self.transducers = pd.read_csv(self.metadata_path / "transducer_configs.csv")
            self.conditions = pd.read_csv(self.metadata_path / "measurement_conditions.csv")
        except FileNotFoundError as e:
            print(f"Warning: Could not load metadata files: {e}")
            print("Some functionality may be limited.")
    
    def load_measurement(self, measurement_file: str) -> Dict:
        """
        Load a single measurement file.
        
        Args:
            measurement_file: Path to HDF5 measurement file
            
        Returns:
            Dictionary containing wavefield data, coordinates, and metadata
        """
        with h5py.File(measurement_file, 'r') as f:
            data = {
                'wavefield': f['wavefield_data'][:],
                'coordinates': f['spatial_coordinates'][:],
                'temporal_info': f['temporal_info'][:],
                'metadata': dict(f.attrs)
            }
        return data
    
    def get_domain_data(self, domain_filter: Dict) -> List[str]:
        """
        Get list of measurement files matching domain criteria.
        
        Args:
            domain_filter: Dictionary specifying domain criteria
                          e.g., {'specimen_type': 'aluminum_plates', 
                                'geometry': 'thickness_5mm'}
        
        Returns:
            List of measurement file paths
        """
        # This is a simplified example - actual implementation would
        # use the metadata tables to filter measurements
        measurement_files = []
        
        # Example logic to find matching files
        for specimen_type in domain_filter.get('specimen_types', ['*']):
            for geometry in domain_filter.get('geometries', ['*']):
                for defect in domain_filter.get('defects', ['*']):
                    pattern = self.data_path / specimen_type / geometry / defect
                    if pattern.exists():
                        measurement_files.extend(pattern.glob("*/*.h5"))
        
        return measurement_files
    
    def create_domain_split(self, source_domains: List[Dict], 
                          target_domains: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Create domain split for transfer learning experiments.
        
        Args:
            source_domains: List of domain specifications for training
            target_domains: List of domain specifications for testing
            
        Returns:
            Tuple of (source_files, target_files)
        """
        source_files = []
        target_files = []
        
        for domain in source_domains:
            source_files.extend(self.get_domain_data(domain))
        
        for domain in target_domains:
            target_files.extend(self.get_domain_data(domain))
        
        return source_files, target_files


def example_domain_shift_evaluation():
    """
    Example: Evaluate domain shift impact on defect detection.
    
    This example demonstrates how to use WaveDS for evaluating
    model performance under domain shift conditions.
    """
    
    # Initialize data loader
    loader = WaveDSLoader("/path/to/WaveDS")
    
    # Define source and target domains
    source_domains = [
        {
            'specimen_types': ['aluminum_plates'],
            'geometries': ['thickness_5mm'],
            'defects': ['hole_2mm', 'hole_5mm']
        }
    ]
    
    target_domains = [
        {
            'specimen_types': ['aluminum_plates'], 
            'geometries': ['thickness_10mm'],  # Different thickness
            'defects': ['hole_2mm', 'hole_5mm']
        },
        {
            'specimen_types': ['steel_plates'],   # Different material
            'geometries': ['thickness_5mm'],
            'defects': ['hole_2mm', 'hole_5mm']
        }
    ]
    
    # Create domain split
    source_files, target_files = loader.create_domain_split(source_domains, target_domains)
    
    print(f"Source domain: {len(source_files)} measurements")
    print(f"Target domains: {len(target_files)} measurements")
    
    # Load and process data (example)
    source_data = []
    for file_path in source_files[:5]:  # Load first 5 for demonstration
        measurement = loader.load_measurement(file_path)
        # Extract features from wavefield data
        features = extract_features(measurement['wavefield'])
        source_data.append(features)
    
    # Train model on source domain
    model = train_defect_detector(source_data)
    
    # Evaluate on target domains
    target_performance = []
    for file_path in target_files[:5]:
        measurement = loader.load_measurement(file_path)
        features = extract_features(measurement['wavefield'])
        performance = evaluate_model(model, features)
        target_performance.append(performance)
    
    # Analyze domain shift impact
    analyze_domain_shift_impact(source_data, target_performance)


def extract_features(wavefield_data: np.ndarray) -> np.ndarray:
    """
    Extract features from ultrasonic wavefield data.
    
    Args:
        wavefield_data: Raw wavefield measurements
        
    Returns:
        Feature vector for machine learning
    """
    # Example feature extraction
    # In practice, this would include sophisticated signal processing
    
    # Time-domain features
    max_amplitude = np.max(np.abs(wavefield_data))
    energy = np.sum(wavefield_data ** 2)
    
    # Frequency-domain features
    fft_data = np.fft.fft(wavefield_data, axis=-1)
    dominant_freq = np.argmax(np.abs(fft_data))
    spectral_centroid = np.sum(np.arange(len(fft_data)) * np.abs(fft_data)) / np.sum(np.abs(fft_data))
    
    # Spatial features
    spatial_variance = np.var(wavefield_data, axis=(0, 1))
    
    features = np.array([max_amplitude, energy, dominant_freq, spectral_centroid] + 
                       spatial_variance.tolist())
    
    return features


def train_defect_detector(training_data: List[np.ndarray]):
    """
    Train a defect detection model.
    
    Args:
        training_data: List of feature vectors
        
    Returns:
        Trained model (placeholder)
    """
    # Placeholder for actual model training
    print("Training defect detection model...")
    return "trained_model"


def evaluate_model(model, features: np.ndarray) -> float:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        features: Feature vector
        
    Returns:
        Performance metric (e.g., accuracy)
    """
    # Placeholder for actual model evaluation
    # In practice, this would return actual performance metrics
    return np.random.random()  # Dummy performance value


def analyze_domain_shift_impact(source_data: List[np.ndarray], 
                               target_performance: List[float]):
    """
    Analyze the impact of domain shift on model performance.
    
    Args:
        source_data: Source domain training data
        target_performance: Performance on target domains
    """
    print("\nDomain Shift Analysis:")
    print(f"Average target performance: {np.mean(target_performance):.3f}")
    print(f"Performance variance: {np.var(target_performance):.3f}")
    
    # Additional analysis would include:
    # - Domain shift quantification metrics
    # - Feature distribution analysis
    # - Visualization of performance degradation


def example_benchmark_evaluation():
    """
    Example: Benchmark evaluation for domain adaptation methods.
    """
    
    print("WaveDS Benchmark Evaluation Example")
    print("="*50)
    
    # Define benchmark scenarios
    scenarios = [
        {
            'name': 'Geometric Domain Shift',
            'description': 'Different specimen thickness',
            'source': {'geometries': ['thickness_5mm']},
            'target': {'geometries': ['thickness_10mm']}
        },
        {
            'name': 'Material Domain Shift', 
            'description': 'Different specimen materials',
            'source': {'specimen_types': ['aluminum_plates']},
            'target': {'specimen_types': ['steel_plates']}
        },
        {
            'name': 'Scale Domain Shift',
            'description': 'Different defect sizes',
            'source': {'defects': ['hole_5mm']},
            'target': {'defects': ['hole_2mm']}
        }
    ]
    
    # Evaluate each scenario
    for scenario in scenarios:
        print(f"\nEvaluating: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        # Here you would:
        # 1. Load source and target domain data
        # 2. Train baseline model on source
        # 3. Evaluate on target (no adaptation)
        # 4. Apply domain adaptation methods
        # 5. Compare performance improvements
        
        print("Baseline (no adaptation): 65.2% accuracy")
        print("Domain adaptation method 1: 78.5% accuracy")
        print("Domain adaptation method 2: 82.1% accuracy")


if __name__ == "__main__":
    print("WaveDS Dataset Usage Examples")
    print("="*50)
    
    # Run example evaluations
    print("\n1. Domain Shift Evaluation Example:")
    example_domain_shift_evaluation()
    
    print("\n2. Benchmark Evaluation Example:")
    example_benchmark_evaluation()
    
    print("\nFor more detailed examples and documentation, see:")
    print("- README.md: Project overview")
    print("- DATASET.md: Detailed dataset documentation") 
    print("- METHODOLOGY.md: Scientific methodology")
    print("- CONTRIBUTING.md: How to contribute")