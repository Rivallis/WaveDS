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
