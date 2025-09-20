
# WaveDS: Ultrasonic Wavefield Domain Shift Dataset

WaveDS is a public dataset specifically designed for investigating domain shift in physics-based measurement data, with a focus on ultrasonic wavefield imaging. This dataset addresses the critical challenge of model robustness when deploying machine learning systems in real-world scenarios where the data distribution differs from the training environment.
# WaveDS
It is an open dataset dedicated to domain shift investigation for ultrasonic wavefield imaging data under various inspection scenarios 

# Description
While domain shift has been extensively studied in computer vision (CV) using benchmark datasets like ImageNet-C and CIFAR10-C, its definition and systematic study in physics-based measurement data remain limited. To address this gap, we developed a dedicated dataset for investigating domain shift in ultrasonic wavefield imaging. Mock specimens with artificially introduced defects were specifically fabricated to create controlled domain shifts in the data. Key measurement factors were systematically varied, including specimen geometry (ranging from flat plates to pipes), defect size, and transducer placement. This dataset supports the quantitative evaluation of model robustness and baseline performance when deployed under domain shift conditions. Additionally, it enables a thorough assessment of the effectiveness of the proposed post-deployment learning scheme in addressing domain shift challenges.

## Overview

This dataset enables quantitative evaluation of model robustness under domain shift and assessment of improvement from post-deployment learning methods. Domain shifts are systematically introduced by varying:

- **Specimen geometry**: Different shapes and configurations of test specimens
- **Defect size**: Various artificial defect dimensions and characteristics  
- **Transducer placement**: Different positioning and orientation of ultrasonic transducers

## Dataset Description

The dataset consists of ultrasonic wavefield imaging data collected from mock specimens with artificial defects. This controlled approach allows for:

- Systematic investigation of domain shift effects
- Quantitative measurement of model performance degradation
- Development and validation of domain adaptation techniques
- Assessment of post-deployment learning strategies

## Key Features

- **Controlled Domain Shifts**: Systematic variation of key parameters affecting ultrasonic measurements
- **Physics-Based Data**: Real ultrasonic wavefield measurements maintaining physical constraints
- **Artificial Defects**: Precisely controlled defect characteristics for reproducible experiments
- **Multiple Scenarios**: Various inspection configurations to simulate real-world deployment variations
- **Quantitative Evaluation**: Enables rigorous assessment of domain adaptation methods

## Applications

This dataset is valuable for researchers and practitioners working on:

- Domain adaptation in non-destructive testing
- Robust machine learning for physics-based measurements
- Transfer learning in ultrasonic imaging
- Post-deployment model adaptation
- Uncertainty quantification in measurement systems

## Data Structure

*[Data structure documentation will be added as the dataset is developed]*

## Usage

*[Usage examples and guidelines will be provided with the dataset release]*

## Installation

*[Installation instructions will be added with code release]*

## Citation

*[Citation information will be provided upon publication]*

## License

This dataset is released under the Attribution-NonCommercial 4.0 International license
=======================================================================
Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible..

## Contributing

*[Contributing guidelines will be added as the project develops]*

## Contact

For questions about the dataset or collaboration opportunities, please open an issue in this repository.
