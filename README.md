# LDGAN for Disease Progression Prediction



## Introduction

This repository contains the implementation of the Longitudinal-Diagnostic Generative Adversarial Network (LDGAN) for predicting disease progression using structural MRI data. Our model aims to generate synthetic MRI images that simulate the progression of a disease over time, alongside predicting clinical scores associated with these future time points.



## Model Overview
LDGAN includes two main components:
- **Generators (G)**: To synthesize future/past patient MRI images from current images.
- **Discriminators (D)**: To distinguish between real and synthetic images.
- **Predictors (P)**: Embedded within the diagnostic generators, tasked with predicting clinical scores.

The network incorporates temporal and output constraints to ensure the synthetic images are temporally coherent and clinically relevant.



## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.8+
- PyTorch 1.8.1+
- NVIDIA GPU with CUDA CuDNN
  

## Installation
Clone the repository and install the required packages:

git clone https://github.com//ldgan.git 

cd ldgan 

pip install -r requirements.txt



## Usage
To train the LDGAN model, run the following command:

python train.py --data_dir <path_to_data> --epochs 50 --batch_size 1

To generate synthetic images and predict clinical scores after training, run:

python predict.py --model_path <path_to_trained_model>



## Data Structure
Your dataset directory should be structured as follows:

data/ 

├── sub1/ 

│ ├── ses1/ 

│ │ └── image.nii 

│ ├── ses2/ 

│ │ └── image.nii 

├── sub2/

 │ ├── ses1/

 │ │ └── image.nii

 │ └── ses2/

│ └── image.nii

## 
## License



## Contact
If you have any questions or feedback, please contact me at 2020213408@bupt.com].



## Acknowledgements
- Adolescent Brain Congnitive Development (ABCD) for providing the dataset.
- This research was supported by Center for Artificial Intelligence in Medical Imaging, BUPT .

