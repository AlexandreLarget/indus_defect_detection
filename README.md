# Industrial defect detection

## Context


Industries are looking for solutions to improve the efficiency of their quality control processess.
Today most of these quality checks are carried out manually. This is time-consuming and leads to many errors.

However, machine learning, and specifically deep learning can bring inovative solutions through computer vision.

As example, we will take the data from ["casting product image data for quality inspection"](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) a Kaggle dataset composed of 1300 images coming from "PILOT TECHNOCAST", a company that casts submersible pump impeller. Each image have 512x512 px dimension.

We will divide the images into train (910 images), validation (195 images) and test (195 images) datasets.

Some random data visualization:

<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/example1.png?raw=true" width="40%" height="40%">
</p>
<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/example2.png?raw=true" width="40%" height="40%">
</p>
<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/example3.png?raw=true" width="40%" height="40%">
</p>


## Method

As we are working with raw data, we first have to preprocess it and make some data augmentation.

After, we will build 3 models based on different architectures and compare them to select the most relevant for our present case.

The 3 models we will work with are VGG16, ResNet50 and VisualTransformer. 
We won't use any transfer learning here.


## Model 0: VGG

We import the VGG16 without top and add 2 dense layers [1024, 512] and 2 dropout layers.

The model early stops after 99 epochs achieved in 430 secondes (GPU Tesla V100).

It scores 99.48% accuracy on the test data.

#### Loss and accuracy curve fromm training


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/vgg_loss_curves.png?raw=true" width="100%" height="100%">
</p>

#### Confusion matrix


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/vgg_cm.png?raw=true" width="50%" height="50%">
</p>


#### Wrong prediction


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/vgg_wrong_pred.png?raw=true" width="20%" height="20%">
</p>


#### Observation

The VGG based model performes extremely well, with only one misclassification (false positive).

The wrongly predicted image doesn't seems to have a defect, so the error might be from the labelling.



## Model 1: ResNet50

We import the ResNet50 without top and add 2 dense layers [1024, 512] and 2 dropout layers.

The model achieves 100 epochs in 410 secondes (GPU Tesla V100).

It scores 97.92% of accuracy on the test data.

#### Loss and accuracy curve fromm training


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/resnet_loss_curves.png?raw=true" width="100%" height="100%">
</p>

#### Confusion matrix


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/resnet_cm.png?raw=true" width="50%" height="50%">
</p>


#### Wrong prediction


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/resnet_wrong_pred.png?raw=true" width="75%" height="75%">
</p>


#### Observation

The ResNet50 based model trains faster than the VGG model but doesn't reach its accuracy score.

With 2 flase positive and 2 false negative on 195 predictions, the model still performs well.



## Model 2: ViT (Visual Transformer)

We create a Visual Transformer model with 4 attention head, 8 layers and patches of [14, 14].

The projected dimension is 64. We add 2 dense layers [1024, 1024] and 2 dropout layers.

The final model counts 17,4 million parameters (26.2 for the ResNet and 15.8 for the VGG).

The model achieves 130 epochs in 280 secondes (more epochs but faster than the 2 others).

It scores 97.92% of accuracy (the exam same score than the model 1).


#### Loss and accuracy curve fromm training


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/vit_loss_curves.png?raw=true" width="100%" height="100%">
</p>

#### Confusion matrix


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/vit_cm.png?raw=true" width="50%" height="50%">
</p>


#### Wrong prediction


<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/vit_wrong_pred.png?raw=true" width="75%" height="75%">
</p>


#### Observation

The ViT reachs a good score, but as the ResNet, it performes worst than the VGG model.

We can try to train the model longer as it doesn't seems to overfit. However, for this case and with the data we have, it seems that the VGG is the model to select.

Here are the compared metrics for the 3 models.

<p align=center>
  <img src="https://github.com/AlexandreLarget/indus_defect_detection/blob/main/images/results.png?raw=true" width="40%" height="40%">
</p>

Thank you




