# My-ML-Projects
Here I will publish my developments in Machine learning area.  

## Repository structure description
Hello dear visitor. What can you find in this repository? 

The first and second files that were committed to this repository contain 
experiments with models from Sklearn. 

These files are named **linear_regression_project.ipynb** and **logistic_regression_project.ipynb** respectively.  

Files that were committed later: **ml_functions.py**, **convolution.py**, **experiments.py**.

The first file contains some Numpy implementations of useful and widespread functions (activations, metrics and losses) in ML.  

The second file contains my implementation of convolution mechanism from scratch using Numpy library. 

Also I added to this file some code for building a padding of the image.

In the file **experiments.py** I wrote some code for conducting experiments over the image with self-builded convolution from previous file.

I applied different kernels of convolution on the image of house. The results of experiments you can find in the data folder.    

File named **vgg_16_project.ipynb** contains my implementation of VGG-16 model for Cifar-100 classification task using Pytorch. 

**XOR_problem.ipynb** and **image_recognition(mnist).ipynb** files contain my code of Tensorflow-builded models for solving a XOR problem and MNIST classification task respectively.

**simple_net.py** file contain Pytorch implemented model (this model is saved in **model.pth** file) for function approximation task 
and **simple_net_valid.py** contain code for validation of builded model.

**classical_ml_models.py** consists of Numpy-code with models from classical ML. At this time, the file contains code for regression models (Linear, Logistic, Multilinear, Multiclass logistic). 

These implementations from scratch using Numpy gives me solid understanding how these models work. 

**segmentation_models.py** file contains models implemented with Pytorch framework for solving semantic segmentation task in **Computer Vision** area. 

At this time I added **U-Net** implementation (with ability to get as an input images of different sizes, not only images of size from original paper) and **Fully Convolutional 
Segmentation Network** that consists only of convolution layers preserving the image dimension with batch normalization and an activation function.
