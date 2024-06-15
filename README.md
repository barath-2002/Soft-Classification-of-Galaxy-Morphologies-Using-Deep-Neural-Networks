# Soft-Classification of Galaxy Morphologies Using Deep Neural Networks

### Overview
This project introduces advancements in galaxy image classification using deep learning techniques. By incorporating constraints into the loss function and leveraging a modified Transformer-based architecture, we aim to emulate human responses in classifying galaxy morphologies accurately. Our work demonstrates that deep learning models, specifically custom-designed Convolutional Neural Networks (CNNs) and Vision Transformers, can effectively classify galaxy images, surpassing traditional methods and crowd-sourced initiatives.

### Introduction
Galaxies exhibit various morphological structures, such as irregular, elliptical, and spiral shapes. Understanding these morphologies is crucial for astrophysical research on galaxy evolution and formation. Traditionally, galaxy classification involved manual observation, but with the exponential increase in the number of images, this method is no longer feasible.

This project leverages deep learning architectures to automatically classify galaxy morphologies from JPEG images. The models are designed to match human classification accuracy, thereby scaling up the classification process while reducing human labor.

### Notebook Content

Below we summarise the contents of each notebook.

- [Parallel_CNN.ipynb](Parallel_CNN.ipynb): Implementation, training, and evaluation of our Parallel CNN network. We also implement Grad-CAM to analyse and interpret model behaviour.

- [VisionTransformer_B16.ipynb](VisionTransformer_B16.ipynb): Implementation, training, and evaluation of our Vision Transformer network.

- [Pretrained_Resnet.ipynb](Pretrained_Resnet.ipynb): Fine-tuning and evaluation of Resnet-50 network that was pre-trained on Imagenet.

- [Pretrained_VGG.ipynb](Pretrained_VGG.ipynb): Fine-tuning and evaluation VGG network that was pre-trained on Imagenet.

- [Constraints_Test.ipynb](Constraints_Test.ipynb): Testing different coefficient values for our custom loss function.

- [Report_Figure_Gen.ipynb](Report_Figure_Gen.ipynb): Generating plots for the report.



### Dataset
Compressed .npz file containing the galaxy images and target distributions is too large to put on github (~500MB)

Please download it from google drive: https://drive.google.com/file/d/1i5f22vNKh_uXY8q-Y_DCxS0gRdbGZZnv/view?usp=sharing

To load the data, use the following code:

```python
# read in image and target datasets
data_path = 'path/to/data'
loaded_arrays = np.load(data_path)

# Retrieve the arrays
train = loaded_arrays['images']
target = loaded_arrays['target']
```
