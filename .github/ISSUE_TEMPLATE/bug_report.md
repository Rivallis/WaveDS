---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 To Reproduce

Steps to reproduce the behavior:

1. Run command '...'
2. With parameters '...'
3. See error

```bash
# Paste the exact command that causes the issue
python TTT_main_MAE.py --model mae_vit_base_patch16 --data_path ./data
```

## ✅ Expected Behavior

A clear and concise description of what you expected to happen.

## 📋 Environment

- **OS**: [e.g., Ubuntu 20.04, macOS Big Sur, Windows 10]
- **Python version**: [e.g., 3.9.7]
- **PyTorch version**: [e.g., 2.0.1]
- **CUDA version** (if applicable): [e.g., 11.8]
- **TTT-MAE version**: [e.g., 1.0.0]

<details>
<summary>Full environment (click to expand)</summary>

```bash
# Please run and paste the output of:
python -c "
import sys, torch, torchvision, timm
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'TIMM: {timm.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
"
```

</details>

## 📄 Error Output

<details>
<summary>Full error traceback (click to expand)</summary>

```
Paste the full error message and traceback here
```

</details>

## 🔧 Additional Context

Add any other context about the problem here. Include:

- Configuration files used
- Data characteristics (if relevant)
- Any modifications made to the code
- Screenshots (if applicable)

## ✅ Checklist

- [ ] I have searched existing issues to make sure this is not a duplicate
- [ ] I have provided all the requested information
- [ ] I have tested with the latest version of TTT-MAE
- [ ] I have tried the basic troubleshooting steps in the documentation
