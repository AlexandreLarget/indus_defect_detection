# Industrial defect detection

## Context


Industries are looking for solutions to improve the efficiency of their quality control processess.
Today most of these quality checks are carried out manually. This is time-consuming and leads to many errors.

However, machine learning, and specifically deep learning can bring inovative solutions through computer vision.

As example, we will take the data from ["casting product image data for quality inspection"](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) a Kaggle dataset composed of 1300 images coming from "PILOT TECHNOCAST", a company that casts submersible pump impeller.

So we will work with real raw data and build different models, with different architecure, and compare them to select the most relevant for our present case.

The 3 models we will work with are VGG16, ResNet50 and VisualTransformer. 
We won't use any transfer learning here.


## Model 0: VGG

We import the VGG16 without top and add it 2 dense layers [1024, 512] and 2 dropout layers.
The model early stops after 99 epochs achieved in 430 secondes (GPU Tesla V100).

Here are the losses abd 
