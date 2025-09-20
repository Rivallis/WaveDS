# TTT-MAE Model and Data Files

This directory contains the pre-trained model and data files for TTT-MAE.

## Files

### Model Files
- `vit-B-ft-US.pt` - Pre-trained Vision Transformer model (655MB)
  - Download from: [Add your model hosting link here]
  - Place in root directory for training/evaluation

### Data Files  
- `USimgV2.mat` - US wavefield dataset (144MB)
  - Download from: [Add your data hosting link here]
  - Place in `data/` directory

## Download Instructions

Due to GitHub file size limitations, large files are hosted separately:

```bash
# Download model file
wget [MODEL_DOWNLOAD_URL] -O vit-B-ft-US.pt

# Download data file
mkdir -p data
wget [DATA_DOWNLOAD_URL] -O data/USimgV2.mat
```

## Alternative: Use Git LFS (Optional)

For development with Git LFS:

```bash
git lfs install
git lfs track "*.pt" "*.mat"
git add .gitattributes
```

## Verification

After downloading, verify the setup:

```bash
python test_verification.py
```

The verification script will check that all required files are present and accessible.
