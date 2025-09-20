# WaveDS Dataset Documentation

## Dataset Overview

The WaveDS (Wavefield Domain Shift) dataset contains ultrasonic wavefield imaging data designed specifically for investigating domain shift phenomena in physics-based measurement systems. The dataset systematically varies key parameters that commonly cause domain shifts in real-world ultrasonic non-destructive testing scenarios.

## Domain Shift Parameters

### 1. Specimen Geometry Variations

The dataset includes measurements from specimens with different geometric configurations:

- **Plate specimens**: Varying thickness, length, and width
- **Curved specimens**: Different curvature radii and orientations
- **Complex geometries**: Multi-layered and irregular shapes
- **Material interfaces**: Different boundary conditions and coupling scenarios

### 2. Defect Size and Characteristics

Artificial defects are systematically varied to create controlled domain shifts:

- **Defect types**: Cracks, holes, inclusions, and delaminations
- **Size variations**: Ranging from sub-wavelength to multiple wavelength dimensions
- **Depth positioning**: Different depths within specimens
- **Orientation**: Various angular orientations relative to the inspection surface

### 3. Transducer Placement and Configuration

Transducer positioning creates domain shifts through:

- **Spatial positioning**: Different x, y coordinates on specimen surface
- **Angular orientation**: Varying tilt and rotation angles
- **Coupling conditions**: Different coupling medium and pressure conditions
- **Array configurations**: Single element vs. array transducers with varying element spacing

## Data Collection Methodology

### Experimental Setup

- **Ultrasonic frequency range**: [To be specified based on actual data]
- **Sampling parameters**: Temporal and spatial resolution details
- **Measurement environment**: Controlled laboratory conditions
- **Equipment specifications**: Transducer types, amplification, and digitization parameters

### Quality Control

- **Calibration procedures**: Regular system calibration protocols
- **Repeatability validation**: Multiple measurements per configuration
- **Noise characterization**: Background noise measurements and SNR analysis
- **Artifact identification**: Systematic identification and documentation of measurement artifacts

## Data Format and Structure

### File Organization

```
WaveDS/
├── data/
│   ├── specimen_type_1/
│   │   ├── geometry_config_1/
│   │   │   ├── defect_size_1/
│   │   │   │   ├── transducer_pos_1/
│   │   │   │   │   ├── measurement_001.h5
│   │   │   │   │   ├── measurement_002.h5
│   │   │   │   │   └── ...
│   │   │   │   └── transducer_pos_N/
│   │   │   └── defect_size_M/
│   │   └── geometry_config_N/
│   └── specimen_type_N/
├── metadata/
│   ├── specimen_specifications.csv
│   ├── defect_parameters.csv
│   ├── transducer_configurations.csv
│   └── measurement_conditions.csv
└── documentation/
    ├── experimental_protocol.md
    ├── data_validation.md
    └── known_issues.md
```

### Data Fields

Each measurement file contains:

- **Wavefield data**: Time-domain ultrasonic measurements
- **Spatial coordinates**: Measurement grid positions
- **Temporal information**: Time stamps and sampling parameters
- **Configuration metadata**: Specimen, defect, and transducer parameters
- **Quality metrics**: Signal-to-noise ratio and measurement confidence

## Benchmark Tasks

The dataset supports several benchmark tasks for domain shift evaluation:

### 1. Defect Detection

- **Task**: Binary classification of defect presence/absence
- **Domain shifts**: Varying specimen geometry and transducer placement
- **Metrics**: Detection accuracy, false positive/negative rates

### 2. Defect Characterization

- **Task**: Regression of defect size and position parameters
- **Domain shifts**: All three parameter categories
- **Metrics**: Mean absolute error, correlation coefficients

### 3. Transfer Learning Scenarios

- **Source domains**: Well-characterized laboratory specimens
- **Target domains**: Modified geometries, defect types, or transducer configurations
- **Evaluation**: Performance degradation and adaptation effectiveness

## Usage Guidelines

### Loading Data

*[Code examples will be provided with dataset release]*

### Domain Split Strategies

Recommended splits for domain shift evaluation:

1. **Geometric domain shift**: Train on simple geometries, test on complex ones
2. **Scale domain shift**: Train on large defects, test on small ones
3. **Position domain shift**: Train on central transducer positions, test on edge positions

### Evaluation Protocols

- **Cross-validation**: Domain-aware splitting to prevent data leakage
- **Baseline methods**: Standard machine learning approaches without domain adaptation
- **Adaptation baselines**: Common domain adaptation techniques for comparison

## Known Limitations

- **Controlled environment**: Laboratory conditions may not capture all real-world variations
- **Artificial defects**: May not fully represent natural defect characteristics
- **Limited materials**: Dataset focuses on specific material types
- **Measurement constraints**: Physical limitations of experimental setup

## Future Extensions

Planned additions to the dataset:

- **Additional material types**: Composites, metals, and ceramics
- **Environmental variations**: Temperature and humidity effects
- **Real defects**: Naturally occurring defects in actual components
- **Multi-modal data**: Combination with other NDT techniques

## Data Validation

### Quality Assurance

- **Signal quality checks**: Automated detection of corrupted measurements
- **Consistency validation**: Cross-checking metadata with measurement parameters
- **Physical plausibility**: Verification against known ultrasonic propagation physics

### Version Control

- **Dataset versioning**: Systematic tracking of data additions and corrections
- **Change logs**: Documentation of all dataset modifications
- **Reproducibility**: Sufficient metadata for measurement reproduction