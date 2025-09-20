# Seq-TTT-MAE Installation Guide

This guide provides comprehensive instructions for setting up the Seq-TTT-MAE (Sequential Test-Time Training with Masked Autoencoders for Ultrasonic Wavefields) environment.

## Quick Start

### Option 1: Using pip (Recommended for most users)

```bash
# Clone the repository
git clone <repository-url>
cd TTT-MAE

# Create virtual environment (recommended)
python -m venv ttt-mae-env
source ttt-mae-env/bin/activate  # On Windows: ttt-mae-env\Scripts\activate

# Install dependencies
pip install -r requirements-minimal.txt
```

### Option 2: Using conda (Recommended for scientific computing)

```bash
# Clone the repository
git clone <repository-url>
cd TTT-MAE

# Create conda environment
conda env create -f environment.yml
conda activate ttt-mae
```

### Option 3: Manual installation

```bash
# Essential packages only
pip install torch>=2.0.0 torchvision>=0.15.0 timm>=0.6.0 numpy>=1.21.0 scipy>=1.7.0 scikit-learn>=1.0.0 matplotlib>=3.5.0 Pillow>=9.0.0 tensorboard>=2.8.0 h5py>=3.7.0 tqdm>=4.60.0
```

## System Requirements

- **Python**: 3.9+ (recommended: 3.9 or 3.10)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8GB RAM, Recommended 16GB+
- **Storage**: At least 2GB free space for dependencies

## GPU Support

### NVIDIA GPUs (CUDA)
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon Macs (MPS)
PyTorch 2.0+ includes MPS support automatically. No additional installation needed.

### AMD GPUs (ROCm)
```bash
# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Verification

Test your installation:

```bash
python -c "
import torch
import torchvision
import timm
import numpy as np
import matplotlib
print('✅ All dependencies installed successfully!')
print(f'PyTorch: {torch.__version__}')
print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"mps\" if torch.backends.mps.is_available() else \"cpu\")}')
"
```

## Common Issues and Solutions

### Issue 1: PyTorch CUDA version mismatch
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: timm version compatibility
```bash
# Install specific timm version if needed
pip install timm==0.6.12
```

### Issue 3: Apple Silicon compatibility
```bash
# Ensure you're using compatible versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue 4: Memory issues
- Reduce batch size in training arguments
- Use gradient accumulation for large effective batch sizes
- Consider using mixed precision training

## Development Setup

For development work:

```bash
# Install additional development dependencies
pip install -r requirements.txt  # Full requirements with dev tools

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Docker Setup (Advanced)

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements-minimal.txt .
RUN pip install -r requirements-minimal.txt

COPY . .
CMD ["python", "TTT_main_MAE.py", "--help"]
```

## Performance Optimization

### For best performance:
1. Use conda for package management when possible
2. Install PyTorch with appropriate GPU support
3. Use the latest compatible versions
4. Consider using mixed precision training (`--mixed_precision`)
5. Optimize batch size for your hardware

## Troubleshooting

If you encounter issues:

1. **Check Python version**: `python --version` (should be 3.9+)
2. **Check PyTorch installation**: `python -c "import torch; print(torch.__version__)"`
3. **Check GPU availability**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Verify all imports**: `python test_verification.py`

## Support

For additional help:
- Check the project documentation
- Run the verification script: `python test_verification.py`
- Review the error messages carefully
- Ensure all dependencies are properly installed

## Version Compatibility

| Component | Minimum | Recommended | Tested |
|-----------|---------|-------------|--------|
| Python | 3.9 | 3.9-3.10 | 3.9.21 |
| PyTorch | 2.0.0 | 2.1+ | 2.6.0 |
| torchvision | 0.15.0 | 0.16+ | 0.21.0 |
| timm | 0.6.0 | 0.6.12+ | 0.6.2 |
| numpy | 1.21.0 | 1.24+ | 1.26.4 |
