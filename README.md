# plant_disease
deep learning project for images classification
This plant diseases image classification is a group project for deep learning developed by Hyunjae Yu and Xiaodan Chen. 

The dataset is from the AI Challenger Competition (https://challenger.ai/competition/pdr2018). The dataset contains 10 plant categories and 61 plant disease classes (by “species-disease-severity”). The dataset is randomly splitted into two sub-datasets: training (31784 images), testing (4540 images). Each image contains one leaf occupying main position of the image.The training set contains the json file of the image and the annotation, and the json file contains each image and the corresponding category ID.

For this project, we just use both the baseline model and the Densenet model, which is released in 2017, to deal with this issue. The latter network used in our project is inspired by the article "Densely Connected Convolutional Networks" and open source implementation. The network and its variants can achieve a very high accuracy among different public image datasets according to this article. In spite of its theoretical accuracy in classification, there are some drawbacks to use DenseNet. DenseNet and its variants mentioned in the article may not feasible for our project due to the computation time limitation. Moreover, they were evaluated with a completely different environment. However, we still believe that it is still a good learning opportunity to just apply such networks to our project.	

There are 3 files in this folder:
--the plant_disease.ipython is developed by transfering learning of Densenet using Pytorch; 

--the baseline model file contains 4 python files using shallow neural network to achieve image classification, although its 35.4% accuracy is not high, we do have a deeper understanding of  the limitation of shallow network. In order to run this model more quickly and easily, we deploy this model on AWS. From this process we learn a lot of configuration methods and the  importance of setting parameters appropriately of an deep learning instance; 

--the project report is the record of the process.

