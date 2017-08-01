#**Traffic Sign Recognition** 

##Files Submitted

Code and running results can be viewed or downloaded [Ipython notebook](https://github.com/shi-ch/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

The writeup report is at [writeup](https://github.com/shi-ch/CarND-Traffic-Sign-Classifier-Project/blob/master/write.md).

The newly added images for test and analysis are under directory [data](https://github.com/shi-ch/CarND-Traffic-Sign-Classifier-Project/blob/master/data/).
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set from the downloaded and unzipped dataset
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
###Writeup / README

####1. Provide a Writeup / README 

You're reading it! 

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Some statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Show all images for all classes to get a rough idea of the images we are dealing with.

Show a summary about how the dataset distrbuted on different classes.

###Design and Test a Model Architecture

####1. The preprocess of training dataset include:

Preserve colors of images, meaning we are not going to use gray scale. 

Brighten some dark pictures. Notice from the previous expration of the dataset that, some pictures seem taken at night thus almost not possible to view even to human eyes.

The technique was used to brighten the pictures is to convert the RGB pictures to HSV(Hue, the relative scale of colors, Saturation, how white is the color, Value, the lightness of color), then increase the V value (lightness) and convert back to RGB scale. As shown below, the brightened images are really easier to view.

[//]: # (Image References)

[image_all]: ./all_classes.png "Figures of all classes"
[image_brightened]: ./brighten_classes.png "Brightened figures of all classes"

![alt text][image_all]
![alt text][image_brightened]

As a last step, I normalized the image data so all the values will be limit to 0 to 1 scale.

I decided to generate additional data because larger training dataset proved to be very useful to increase the accuracy when tested with validation datasets. 

To add more data to the the data set, I rotated each image twice by slightly(and randomly) rotate each image to two opposite directions, thus I made the training dataset three times larger.  

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final architecture is based on leNet. 

* Convolutional. Input = 32x32x3. Output = 28x28x12. pad='VALID'
* Pooling. Input = 28x28x12. Output = 14x14x12.
* Convolutional. Input = 14x14x12. Output = 10x10x32. pad='VALID'
* Pooling. Input = 10x10x32. Output = 5x5x32.
* Flatten (800)
* Full Cononected. Input = 800. Output = 120.
* Full Cononected. Input = 120. Output = 84.
* Full Cononected. Input = 84. Output = 43.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The optimizer used in the training is AdamOptimizer with default exponential decay rates. Ran some training with AdagradOptimizer and no siginificant difference observed.

The default epochs for training is 10, since now the network become deeper since I increased the depth of two layers, notice that the neural network can reach a stable percent number on validate dataset around epoch 15 - 20.

The batch size was set from default value of 128 to 64, the reason was to limit the memory usage on the linux VM hosted on my laptop.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.6%
* test set accuracy of 93.2%

* What was the first architecture that was tried and why was it chosen?
The process is iterative approach to explore how different parameters' changes could have impact on accuracy.

Initially letNet was chosen because it showed very good performance when handling similar size(28x28 vs. 32x32) of pictures and similar size of training samples. Run inital model result was ~80% accuracy on validation dataset. So this seemed a not bad starting point.

* What were some problems with the initial architecture?
The initial architecture was design to handle gray images thus the volume was not big enough to take in much more complex(colorful) images with much different background and different light conditions and more complex pictures(e.g. bike, car, slippery, 'stop' etc.).

As to preprocessing, since traffic signs have color elements. And actually color is important when people design traffic signs. So intuitely we should keep RGB color information in preprocessing. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Since I noticed that running more epochs can bring certain improvements on accuracy, So I extended training epochs to 40, 80, 400. However all stuck somewhere away from 93%.

So first focus on preprocessing inputs by not using gray images, then increase training datasets by randomly rotating the images a tiny angle to generate new images. Then brigtening the dark images. These efforts all brought some small improvements. But the impact is not big enough to consistently staying above 93% accuracy.

Observed that the entropy of input actually is 3 times than gray images now, I also increased the volume of each layers(parameter numbers) about 2 -3 times. These adjustments proved to be critical to the accuracy improvement.


Then finalize the training epoch after these experiements. Since for now, the epoch number usually could be set to less than 20. with this epcoh number, the concern of overfitting did not raise.

* Which parameters were tuned? How were they adjusted and why?

Epochs, batch size, depths of Convolutional layers' filters. While depths were increased, the batch sizes were decreased.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

* What architecture was chosen?

leNet architecture was chosen.

* Why did you believe it would be relevant to the traffic sign application?

It worked for similar size of pictures and datasets.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

For valid dataset and test dataset, the accuracy is >93%, for the pictures I captured from google maps' street view, the final model can still give 80% accuracy. If consider one image is not actually different than training data, the actual accuracy is around ~90%. This proved the model was working quite well.

###Test a Model on New Images

####1. Choose ten German traffic signs found on the google maps in Munich city and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

As discussed in notebook, one picture did not get recognized was due to the shape of the sign are different than the picture in training dataset. If this was excluded, the accuracy on these newly downloaded images is close to 90% too.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set.

####3. Describe how certain the model is when predicting on each of the ten new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The softmax, top 5 guesses numbers and correspondent dicussions could be found in notebook. Other than the sign with changed shape, for another image that the model made mistake, there are two guesses have very close possibilities(Spped Limit 50 and Speed Limit 30).  



