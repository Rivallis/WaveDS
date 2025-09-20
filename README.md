
# Mitigating Domain Shift in Ultrasonic Wavefield Pattern Analysis through Test-Time Training

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-ICASSP%202026-red.svg)](#citation)

This repository contains the official implementation of **Mitigating Domain Shift in Ultrasonic Wavefield Pattern Analysis through Test-Time Training** for ICASSP 2026 submission. Our method addresses the critical challenge of domain shift in physics-based signal analysis by enabling deployed neural networks to adapt on-the-fly during inference.



## 🎯 Overview

Real-world ultrasonic inspection environments often exhibit statistical discrepancies from training data due to variations in:
- **Specimen structures** (flat plates vs. pipes)
- **Transducer positions** (top, side, left/right placement)  
- **Defect conditions** (sizes)

Our **Sequential TTT-MAE** approach enables progressive feature adaptation during inference, achieving significant performance improvements:

| Domain Shift | Deployed Model | **Seq-TTT-MAE (Ours)** | Improvement |
|--------------|----------------|------------------------|-------------|
| Type-A       | 54.45%         | **87.50%**             | +33.05%     |
| Type-B       | 66.46%         | **81.15%**             | +14.69%     |
| Type-C       | 56.09%         | **76.77%**             | +20.68%     |

## 🚀 Key Features

### ✨ **Sequential Test-Time Training**
- Exploits temporal structure in ultrasonic wavefield snapshots
- Progressive adaptation through mini-batch processing
- Self-supervised learning with Masked Autoencoder (MAE)

### 📊 **Benchmarking Dataset**  
## Overview
- **Specimen geometry**: Different shapes and configurations of test specimens
- **Defect size**: Various artificial defect dimensions and characteristics  
- **Transducer placement**: Different positioning and orientation of ultrasonic transducers

## Dataset Description

The dataset consists of ultrasonic wavefield imaging data collected from mock specimens with artificial defects. This controlled approach allows for:

- Systematic investigation of domain shift effects
- Quantitative measurement of model performance degradation
- Development and validation of domain adaptation techniques
- Assessment of post-deployment learning strategies

## 📦 Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Rivallis/WaveDS.git
cd WaveDS

# Install dependencies
pip install -r requirements.txt

# Download large files (model and data)
# See LARGE_FILES.md for download instructions

# Verify installation
python test_verification.py
```

### Alternative Installation Methods

**Conda Environment:**
```bash
conda env create -f environment.yml
conda activate ttt-mae
```

**Minimal Installation:**
```bash
pip install -r requirements-minimal.txt
```

## 🚀 Usage

### Basic Example

```python
import torch
from TTT_main_MAE import create_model, run_sequential_ttt

# Load pre-trained model
model = create_model('mae_vit_base_patch16', pretrained=True)

# Load your ultrasonic wavefield data
# wavefield_sequence: Sequential snapshots from ultrasonic inspection
# labels: Ground truth defect/non-defect labels

# Apply Sequential TTT-MAE
adapted_model, results = run_sequential_ttt(
    model=model,
    test_data=wavefield_sequence,
    batch_size=32,
    adaptation_steps=32,
    learning_rate=1e-3
)

# Make predictions with adapted model
predictions = adapted_model(new_wavefield_data)
```

### Advanced Usage

**Custom Domain Shift Scenarios:**
```python
from engine_TTT_wavefield_LN_MAE_vis import TTTEngine

# Initialize TTT engine
ttt_engine = TTTEngine(
    model=model,
    auxiliary_task='mae',
    mask_ratio=0.75,
    temporal_grouping=True
)

# Adapt to specific domain shift
ttt_engine.adapt_to_domain(
    source_data=training_data,
    target_data=test_data,
    domain_type='transducer_position'  # Type-A, Type-B, or Type-C
)
```

## 📊 Experimental Results

### Domain Shift Performance

| Method | Type-A (Acc.) | Type-B (Acc.) | Type-C (Acc.) | Avg. Improvement |
|--------|---------------|---------------|---------------|------------------|
| Deployed Model | 54.45% | 66.46% | 56.09% | Baseline |
| LayerNorm Adaptation | 61.35% | 62.40% | 59.38% | +2.76% |
| TTT-MAE | 69.90% | 63.33% | 61.82% | +6.09% |
| **Seq-TTT-MAE (Ours)** | **87.50%** | **81.15%** | **76.77%** | **+22.81%** |

### Key Findings

1. **Temporal Structure Matters**: Sequential processing of wavefield snapshots significantly improves adaptation
2. **Hyperparameter Sensitivity**: Optimal performance at batch size 32 with 32 adaptation iterations  
3. **Physics-Aware Design**: Custom adaptation for ultrasonic data outperforms general CV methods

## 🔬 Technical Details

### Architecture

- **Backbone**: ViT-Base with Masked Autoencoder (MAE)
- **Pre-training**: ImageNet pre-trained ViT-MAE
- **Fine-tuning**: Source dataset (aluminum plate specimens)
- **Adaptation**: Sequential TTT with temporal batching

### Algorithm Overview

```
Input: Sequential wavefield snapshots X_T = {x_t}_{t=1}^T
Output: Adapted model parameters θ*

1. Group snapshots into temporal mini-batches B_1, ..., B_{T_B}
2. For each mini-batch B_t:
   a. Apply MAE auxiliary task with masking
   b. Update encoder parameters θ_{(t)}
   c. Maintain temporal correlations
3. Use adapted model for classification
```

## 📈 Dataset Specifications

### Source Dataset (Training)
- **Specimen**: Aluminum plates  
- **Images**: 7,004 samples (224×224×3)
- **Labels**: Binary (defect/non-defect)

### Target Dataset (Testing)  
- **Specimen**: Aluminum pipes
- **Images**: 4,008 samples (122×301×3)
- **Domain Shifts**: 3 types based on transducer position

| Domain Type | Transducer Position | Specimens | Samples |
|-------------|-------------------|-----------|---------|
| Type-A | Top (middle) | 6 | 1,002 |
| Type-B | Side | 6 | 1,002 |  
| Type-C | Top (left/right) | 12 | 2,004 |

*[Usage examples and guidelines will be provided with the dataset release]*

## 📄 License

This project is released under the [MIT License](LICENSE), making it freely available for research and commercial use.

## 🤝 Contributing

If you want to contribute to this project, please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- Extended domain shift scenarios
- Additional TTT methods comparison  
- Improved visualization tools
- Performance optimizations
- Documentation improvements

## 📞 Contact

- **Authors**: Jiaxing Ye, Takumi Kobayashi
- **Affiliation**: National Institute of Advanced Industrial Science and Technology (AIST)
- **Issues**: Please open an issue for questions or bug reports
- **Collaboration**: Contact through GitHub for research collaboration opportunities

## 🙏 Acknowledgments

- ImageNet pre-trained MAE models from Facebook Research
- Ultrasonic wavefield imaging source dataset contributors
- PyTorch and timm library maintainers
- Open-source community for development tools and frameworks
