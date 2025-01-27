# Neural Style Transfer Project
This project explores Neural Style Transfer (NST), a deep learning technique that transfers the style of one image to another while preserving its content. Three different architectures are used: Vanilla NST, Residual NST, and Multi-Scale NST.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architectures Used](#architectures-used)
   - [Vanilla NST](#vanilla-nst)
   - [Residual NST](#residual-nst)
   - [Multi-Scale NST](#multi-scale-nst)
3. [Key Features](#key-features)
4. [Setup and Installation](#setup-and-installation)
   - [Dependencies](#dependencies)
   - [Installation Instructions](#installation-instructions)
5. [How It Works](#how-it-works)
   - [Preprocessing](#preprocessing)
   - [Training the Model](#training-the-model)
   - [Loss Functions](#loss-functions)
   - [Optimization](#optimization)
6. [Running the Model](#running-the-model)
   - [Training](#training)
   - [Testing](#testing)
   - [Visualizing Results](#visualizing-results)

---

## Project Overview
The goal of this project is to implement NST using different architectures and compare their effectiveness in transferring artistic styles to content images.

## Architectures Used

### Vanilla NST
This is the basic implementation of Neural Style Transfer, where convolutional layers are used to extract features from both content and style images. The model attempts to generate a new image that preserves the content of the content image while transferring the style from the style image.

### Residual NST
This architecture includes residual connections between layers of the neural network. Residual connections help with the vanishing gradient problem and enable the model to learn better, especially for deeper networks.

### Multi-Scale NST
This approach uses multi-scale convolution to capture style information at different resolutions. This enables the model to preserve finer details in the generated image by processing the image at various scales.

## Key Features
- **Preprocessing and Deprocessing**: Converting images into the appropriate format for training.
- **Gram Matrix**: Used to calculate the style loss by measuring the similarity between the style image and the generated image.
- **Loss Function**: Combines content loss, style loss, and variation loss to optimize the generated image.
- **Optimizer**: Adam optimizer is used for updating the weights during the training process.

## Setup and Installation

### Dependencies
- Python 3.x
- TensorFlow / PyTorch
- NumPy
- OpenCV
- Matplotlib

### Installation Instructions
1. Clone the repository:  
   `git clone https://github.com/your-username/neural-style-transfer.git`
2. Install dependencies:  
   `pip install -r requirements.txt`

## How It Works

### Preprocessing
The content and style images are loaded, resized to a standard resolution (e.g., 512x512), and converted into the appropriate format for neural network processing.

### Training the Model
The model is trained by minimizing a combined loss function that includes content loss, style loss, and variation loss.

### Loss Functions
- **Content Loss**: Measures the difference in content between the content image and the generated image.
- **Style Loss**: Measures the difference in the artistic style between the style image and the generated image using Gram matrix.
- **Variation Loss**: Ensures the generated image does not contain unnecessary noise.

### Optimization
The model uses the Adam optimizer to minimize the loss function during training.

## Running the Model

### Training
You can train the model by running the training script with the following command:
```bash
python train.py --content_image path/to/content_image --style_image path/to/style_image
