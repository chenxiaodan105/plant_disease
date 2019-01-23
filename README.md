# plant_disease
deep learning project for images classification
This plant diseases image classification is a group project for deep learning developed by Hyunjae Yu and Xiaodan Chen. 

The dataset is from the AI Challenger Competition (https://challenger.ai/competition/pdr2018). The dataset contains 10 plant categories and 61 plant disease classes (by “species-disease-severity”). The dataset is randomly splitted into two sub-datasets: training (31784 images), testing (4540 images). Each image contains one leaf occupying main position of the image.The training set contains the json file of the image and the annotation, and the json file contains each image and the corresponding category ID.

For this project, we just use both the baseline model and the Densenet model, which is released in 2017, to deal with this issue. The latter network used in our project is inspired by the article "Densely Connected Convolutional Networks" and open source implementation. The network and its variants can achieve a very high accuracy among different public image datasets according to this article. 

There are 3 files in this folder:

--the plant_disease.ipython is developed by the transfering learning of Densenet using Pytorch; 

--the baseline model file contains 4 python files using shallow neural network using tensorflow to achieve image classification

--the project report is the record of the process.

