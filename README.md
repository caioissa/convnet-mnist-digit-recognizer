# Convnet MNIST Digit Recognizer
### Convolutional Network to identify MNIST numeric handwritten digits
* The model achieves 97% accuracy in 10k validation samples with a single epoch.
* It uses CUDA with GPU is available
## Installation
Create a virtualenv and run
```bash
pip install -r requirements.txt
```
## Extract Data
The train script will download the data if it is not already downloaded via torchvision
## Usage
From root folder:
1. Run to train the network. The parameter refers to the number of epochs. *Please note that, unless using GPU, the epochs take quite some time and 1 or 2 are usually enough*.
```bash
python src/train.py 1
```
The model will be saved in `model/model.pytorch`
2. Run if you want some visual validation (just for fun) 
```bash
python src/display_predictions.py
```
