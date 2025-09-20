# WaveDS Methodology: Systematic Domain Shift Investigation

## Introduction

The WaveDS dataset methodology is designed to systematically investigate domain shift phenomena in ultrasonic wavefield imaging through controlled parameter variations. This document outlines the scientific approach used to create meaningful domain shifts while maintaining measurement quality and physical validity.

## Domain Shift Framework

### Definition of Domain Shift in Ultrasonic Imaging

In ultrasonic wavefield imaging, domain shift occurs when the statistical properties of the measurement data change between training and deployment scenarios. This can result from:

1. **Physical parameter changes**: Specimen properties, defect characteristics, or measurement setup
2. **Environmental variations**: Temperature, coupling conditions, or ambient noise
3. **Equipment differences**: Transducer variations, system calibration, or signal processing

### Systematic Parameter Variation Strategy

The WaveDS methodology creates controlled domain shifts by systematically varying three key parameter categories:

#### 1. Specimen Geometry Domain Shifts

**Rationale**: Geometric variations directly affect ultrasonic wave propagation paths, reflection patterns, and mode conversion phenomena.

**Implementation**:
- **Baseline geometry**: Standard rectangular plate specimens with well-characterized properties
- **Variations**: Systematic changes in thickness, curvature, edge conditions, and surface roughness
- **Parameterization**: Quantitative geometric parameters enabling controlled interpolation and extrapolation

**Expected impact**: Changes in wave propagation time, reflection coefficients, and modal dispersion characteristics.

#### 2. Defect Size and Type Domain Shifts

**Rationale**: Defect characteristics are the primary target for ultrasonic inspection, making size and type variations critical for robustness evaluation.

**Implementation**:
- **Controlled defect creation**: Precisely manufactured artificial defects with known dimensions
- **Size progression**: Logarithmic scaling from sub-wavelength to multiple wavelength dimensions
- **Type variations**: Different defect morphologies (cracks, holes, inclusions, delaminations)
- **Positioning control**: Systematic depth and lateral position variations

**Expected impact**: Changes in scattering amplitude, frequency dependence, and directivity patterns.

#### 3. Transducer Placement Domain Shifts

**Rationale**: Transducer positioning affects signal amplitude, coverage area, and measurement sensitivity, commonly varying between inspections.

**Implementation**:
- **Spatial sampling**: Grid-based positioning with known coordinate systems
- **Angular variations**: Systematic tilt and rotation angle changes
- **Coupling variations**: Different pressure and medium conditions
- **Array configurations**: Single element vs. array measurements with varying apertures

**Expected impact**: Changes in signal-to-noise ratio, beam focusing effects, and coverage uniformity.

## Experimental Design Principles

### 1. Factorial Design Approach

The dataset employs a structured factorial design to ensure systematic coverage of the parameter space:

- **Full factorial**: Complete combinations of key parameter levels
- **Fractional factorial**: Efficient sampling for high-dimensional parameter spaces
- **Orthogonal sampling**: Independent variation of parameters to enable statistical analysis

### 2. Physical Validity Constraints

All parameter variations maintain physical validity through:

- **Physics-based bounds**: Parameter ranges consistent with ultrasonic propagation physics
- **Measurement feasibility**: Configurations achievable with available instrumentation
- **Signal quality**: Sufficient signal-to-noise ratio for meaningful analysis

### 3. Statistical Considerations

- **Replication**: Multiple measurements per configuration to assess measurement uncertainty
- **Randomization**: Random measurement order to minimize systematic bias
- **Blocking**: Grouping measurements to control for time-dependent variations

## Mock Specimen Design

### Artificial Defect Creation

**Objectives**:
- Precise control of defect parameters
- Reproducible defect characteristics
- Known ground truth for validation

**Manufacturing methods**:
- **Machined defects**: Precise geometric control using CNC machining
- **Additive manufacturing**: Complex internal geometries through 3D printing
- **Assembly techniques**: Controlled interfaces and delaminations
- **Material insertion**: Inclusion of foreign materials with known properties

### Specimen Materials

**Selection criteria**:
- **Well-characterized properties**: Known elastic constants and attenuation
- **Measurement compatibility**: Suitable for ultrasonic inspection
- **Manufacturing feasibility**: Amenable to controlled defect creation
- **Research relevance**: Representative of practical inspection scenarios

## Measurement Protocol

### Data Acquisition Standards

**Temporal parameters**:
- **Sampling rate**: Sufficient for accurate waveform capture (typically >10× center frequency)
- **Record length**: Adequate for complete wavefield evolution
- **Averaging**: Multiple acquisitions to improve signal-to-noise ratio

**Spatial parameters**:
- **Grid density**: Sufficient spatial resolution for wavefield reconstruction
- **Coverage area**: Complete specimen coverage with appropriate boundaries
- **Registration accuracy**: Precise spatial coordinate determination

### Quality Control Procedures

**Real-time monitoring**:
- **Signal amplitude tracking**: Continuous monitoring of signal levels
- **Coupling assessment**: Verification of transducer-specimen contact
- **Environmental monitoring**: Temperature, humidity, and vibration tracking

**Post-acquisition validation**:
- **Waveform integrity**: Automated detection of corrupted or clipped signals
- **Consistency checks**: Verification of measurement repeatability
- **Physical plausibility**: Comparison with expected ultrasonic behavior

## Domain Shift Quantification

### Metrics for Domain Shift Assessment

**Statistical measures**:
- **Distribution divergence**: Kullback-Leibler divergence, Wasserstein distance
- **Feature-level changes**: Principal component analysis, domain discriminator accuracy
- **Performance degradation**: Model accuracy reduction across domains

**Physics-based measures**:
- **Wave propagation changes**: Time-of-flight variations, dispersion characteristics
- **Scattering parameter changes**: Amplitude and phase variations
- **Signal quality metrics**: Signal-to-noise ratio, coherence measures

### Benchmark Protocols

**Standard evaluation procedures**:
- **Cross-domain validation**: Training on source domain, testing on target domain
- **Progressive domain shift**: Gradual parameter changes to assess transition behavior
- **Multi-domain generalization**: Training on multiple source domains for robustness

## Validation and Verification

### Measurement Validation

**Physical consistency**:
- **Reciprocity verification**: Bidirectional measurement consistency
- **Energy conservation**: Verification of energy balance in measurements
- **Dispersion validation**: Comparison with theoretical dispersion curves

**Numerical validation**:
- **Finite element modeling**: Comparison with simulated wavefields
- **Analytical solutions**: Validation against known analytical cases
- **Literature comparison**: Consistency with published results

### Dataset Integrity

**Quality assurance**:
- **Metadata validation**: Consistency between recorded and actual parameters
- **File integrity**: Verification of data file completeness and format
- **Version control**: Systematic tracking of dataset versions and changes

## Applications to Machine Learning

### Domain Adaptation Benchmarks

The systematic parameter variations enable specific domain adaptation scenarios:

- **Unsupervised domain adaptation**: No target domain labels
- **Few-shot learning**: Limited target domain samples
- **Online adaptation**: Sequential learning from target domain data

### Robustness Evaluation

**Stress testing**:
- **Worst-case scenarios**: Maximum parameter deviations
- **Gradual degradation**: Progressive domain shift assessment
- **Failure mode analysis**: Identification of critical failure points

## Future Methodology Extensions

### Advanced Domain Shift Scenarios

- **Temporal domain shifts**: Equipment aging and degradation effects
- **Multi-physics coupling**: Combined thermal, mechanical, and ultrasonic effects
- **Real-world deployment**: Field measurement scenarios with uncontrolled variations

### Enhanced Characterization

- **Uncertainty quantification**: Probabilistic assessment of domain shift effects
- **Sensitivity analysis**: Parameter importance ranking
- **Interaction effects**: Higher-order parameter interactions

This methodology provides a rigorous framework for investigating domain shift in ultrasonic wavefield imaging while maintaining scientific rigor and practical relevance for machine learning applications.