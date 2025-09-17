# WaveDS Data Structure Example

This directory demonstrates the proposed data organization structure for the WaveDS dataset.

## Directory Hierarchy

```
WaveDS/
├── data/                           # Main data directory
│   ├── aluminum_plates/            # Specimen type 1
│   │   ├── thickness_5mm/          # Geometry configuration 1
│   │   │   ├── hole_2mm/           # Defect size 1
│   │   │   │   ├── center_normal/  # Transducer position 1
│   │   │   │   ├── edge_15deg/     # Transducer position 2
│   │   │   │   └── corner_45deg/   # Transducer position 3
│   │   │   ├── hole_5mm/           # Defect size 2
│   │   │   └── crack_10mm/         # Defect size 3
│   │   ├── thickness_10mm/         # Geometry configuration 2
│   │   └── curved_50mm_radius/     # Geometry configuration 3
│   ├── steel_plates/               # Specimen type 2
│   └── composite_panels/           # Specimen type 3
├── metadata/                       # Metadata and documentation
│   ├── specimen_specifications.csv # Detailed specimen parameters
│   ├── defect_parameters.csv       # Defect characteristics
│   ├── transducer_configs.csv      # Transducer setup details
│   ├── measurement_conditions.csv  # Environmental and setup conditions
│   └── data_validation_report.pdf  # Quality assurance documentation
├── tools/                          # Analysis and utility tools
│   ├── data_loaders/               # Data loading utilities
│   ├── visualization/              # Visualization tools
│   ├── domain_shift_metrics/       # Domain shift quantification
│   └── benchmarks/                 # Benchmark evaluation scripts
└── documentation/                  # Extended documentation
    ├── experimental_protocols.md   # Detailed measurement procedures
    ├── calibration_procedures.md   # Equipment calibration protocols
    ├── troubleshooting.md          # Common issues and solutions
    └── changelog.md                # Dataset version history
```

## File Naming Convention

### Measurement Data Files
- Format: `measurement_{specimen}_{geometry}_{defect}_{transducer}_{replicate}.h5`
- Example: `measurement_aluminum_5mm_hole2mm_center_001.h5`

### Metadata Files
- Descriptive names indicating content and purpose
- CSV format for tabular data, JSON for structured metadata
- Consistent field naming across all metadata files

## Data File Contents

Each measurement file (HDF5 format) contains:

### Primary Data Groups
- `/wavefield_data`: Time-domain measurement arrays
- `/spatial_coordinates`: Measurement grid positions
- `/temporal_info`: Time stamps and sampling parameters
- `/metadata`: Configuration and quality information

### Attributes
- Measurement timestamp
- Equipment identification
- Calibration references
- Data quality metrics
- Processing parameters

## Metadata Structure

### Specimen Specifications (`specimen_specifications.csv`)
- `specimen_id`: Unique identifier
- `material_type`: Material classification
- `geometry_type`: Geometric configuration
- `dimensions`: Physical dimensions
- `material_properties`: Elastic constants, density, etc.

### Defect Parameters (`defect_parameters.csv`)
- `defect_id`: Unique identifier
- `specimen_id`: Reference to specimen
- `defect_type`: Classification (hole, crack, etc.)
- `dimensions`: Size parameters
- `position`: Location within specimen
- `orientation`: Angular orientation

### Transducer Configurations (`transducer_configs.csv`)
- `config_id`: Unique identifier
- `transducer_type`: Equipment model and specifications
- `position`: Spatial coordinates
- `orientation`: Angular positioning
- `coupling_info`: Coupling medium and conditions

### Measurement Conditions (`measurement_conditions.csv`)
- `measurement_id`: Unique identifier
- `timestamp`: Measurement time
- `temperature`: Environmental temperature
- `humidity`: Relative humidity
- `equipment_status`: Calibration and health status

## Quality Assurance

### Data Validation
- Automated checks for file integrity
- Physical plausibility validation
- Consistency verification across metadata
- Signal quality assessment

### Version Control
- Systematic versioning of dataset releases
- Change logs documenting modifications
- Backward compatibility maintenance
- Data provenance tracking

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