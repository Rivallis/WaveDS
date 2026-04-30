# WaveDS Dataset Documentation

## Dataset Overview

The WaveDS (Wavefield Domain Shift) dataset contains ultrasonic wavefield imaging data designed specifically for investigating domain shift phenomena in physics-based measurement systems. The dataset systematically varies key parameters that commonly cause domain shifts in real-world ultrasonic non-destructive testing scenarios.

## Domain Shift Parameters
### 1. Specimen Geometry Variations
The dataset includes measurements from specimens with different geometric configurations:
- **Specimens characteristics**: Varying thickness and width
- **Variation in geometries**: Flat plate and curved pipes
- **Material interfaces**: Different boundary conditions and coupling scenarios

### 2. Defect Size and Characteristics
Artificial defects are systematically varied to create controlled domain shifts:
- **Defect types**: Slit cracks and drilled holes
- **Size variations**: Ranging from 1mm to 3mm in size
- **Depth positioning**: Full penetrated and half-penetrated cases

### 3. Transducer Placement and Configuration
Transducer positioning creates domain shifts through:
- **Spatial positioning**: Different x, y coordinates on specimen surface
- **Angular orientation**: Varying tilt and rotation angles

# WaveDS Data Structure Example
This directory demonstrates the proposed data organization structure for the WaveDS dataset.

## Directory Hierarchy

```
WaveDS/
├── examples/                           # Main data directory
│   ├── PlateWavefields/            # Specimen type 1, plate-shaped; used for model training (ViT finetuning)
│   │   ├── Various wavefield snapshots, including both defective and normal patterns of wave propagation. 
│   ├── PipeWavefields/               # Specimen type 2, pipe-shaped; used for test-time training at inference stage, with different sensor positioning
│   │   ├── Edge/
│   │   ├── Top/
│   │   ├── Top_LR/
Various wavefield snapshots, including both defective and normal patterns of wave propagation. 
```
The USimgAIST dataset can be downloaded via following URL:
https://drive.google.com/drive/folders/1P5k5RWfNcdmg8JXEeoeesvbjDB8XrZiX?usp=sharing

## File Naming Convention
-No. of specimen (#)
-size of defect-penetrated (Kanttu)/non-penetrated (non-Kanttu) cases 
-size of defect-
-laser scanning frequency
-transducer incident angle

### Measurement Data Files
- Format: `*.jpg`
- Example: `N27-3mmnonKanttu_3_1_90_500_45.jpg`

This structure provides systematic organization enabling efficient data access, clear domain separation, and comprehensive metadata tracking for robust domain shift research.

## Benchmark Tasks

The dataset supports several benchmark tasks for domain shift evaluation:

### 1. Defect Detection

- **Task**: Binary classification of defect presence/absence
- **Domain shifts**: Varying specimen geometry and transducer placement
- **Metrics**: Detection accuracy, false positive/negative rates

### 2. Transfer Learning Scenarios

- **Source domains**: Data collected from plate specimens 
- **Target domains**: Data captured with specimens of modified geometries (pipes), defect types, or transducer configurations
- **Evaluation**: Performance degradation and adaptation effectiveness

### Evaluation Protocols

- **Cross-validation**: Domain-aware splitting to prevent data leakage
- **Baseline methods**: Standard finetuning approaches without proposed test-time domain adaptation
- **Adaptation baselines**: Common domain adaptation techniques for comparison
