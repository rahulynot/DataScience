#### Transfer any artistic style to your image using Convolution Neural Networks

**Result:**


![alt text](https://github.com/rahulynot/NeuralStyleTransfer/blob/master/Images/StyleTransferedImage.png)

**Content Image:**
![alt text](https://github.com/rahulynot/NeuralStyleTransfer/blob/master/Images/Berlin.jpg)

**Style Image:**
![alt text](https://github.com/rahulynot/NeuralStyleTransfer/blob/master/Images/JapaneseArt.jpg)


**Requirements:**
1. Tensorflow-gpu: 1.13
2. Pretrained imagenet model (https://www.kaggle.com/teksab/imagenetvggverydeep19mat)

**Usage:**

python NeuralStyleTransfer.py
--content-image 'path/to/content_image.jpg' 
--style-image 'path/to/style_image.jpg' 
--pretrained-model 'path/to//imagenet-vgg-verydeep-19.mat'

On GPU Quadro P1000 takes about 10 minutes to run 1000 epochs.

(default number of epochs are 1000. Can be changed with argument --epochs)
