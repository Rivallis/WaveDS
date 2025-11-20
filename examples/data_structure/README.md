# WaveDS Data Structure Example

This directory demonstrates the proposed data organization structure for the WaveDS dataset.

## Directory Hierarchy

```
WaveDS/
├── examples/                           # Main data directory
│   ├── PlateWavefields/            # Specimen type 1, plate-shaped; used for model training (ViT finetuning)
│   │   ├── Various wavefield snapshots, including both defective and normal patterns of wave propagation. 
│   ├── PipeWavefields/               # Specimen type 2, pipe-shaped; used for test-time training at inference stage
```

## File Naming Convention
-No. of specimen (#)
-size of defect-penetrated (Kanttu)/non-penetrated (non-Kanttu) cases 
-size of defect-
-laser scanning frequency
-transducer incident angle

### Measurement Data Files
- Format: `*.jpg`
- Example: `N27-3mmnonKanttu_3_1_90_500_45.jpg`

## Data File Contents
Each measurement file (HDF5 format) contains:


## Usage Examples

### Loading Data
```python
# Example Python code for data loading
import h5py
import pandas as pd

# Load measurement data
with h5py.File('measurement_aluminum_5mm_hole2mm_center_001.h5', 'r') as f:
    wavefield = f['wavefield_data'][:]
    coordinates = f['spatial_coordinates'][:]
    metadata = dict(f.attrs)

# Load metadata
specimens = pd.read_csv('metadata/specimen_specifications.csv')
defects = pd.read_csv('metadata/defect_parameters.csv')
```

### Domain Split Example
```python
# Example domain split for evaluation
source_specimens = ['aluminum_plates/thickness_5mm']
target_specimens = ['aluminum_plates/thickness_10mm', 'steel_plates/thickness_5mm']

# Train on source domain, evaluate on target domains
train_data = load_domain_data(source_specimens)
test_data = load_domain_data(target_specimens)
```

This structure provides systematic organization enabling efficient data access, clear domain separation, and comprehensive metadata tracking for robust domain shift research.
