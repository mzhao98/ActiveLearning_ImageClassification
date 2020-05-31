# ActiveLearning_ImageClassification
Active Learning for Multiclass Image Classification on Fruits360 Dataset. This work was done for Caltech's CS 186 Computer Vision course with Prof. Pietro Perona. The link to the associated report is https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/active_learning_paper.pdf.

## What is Active Learning?
Active learning is a machine learning framework in which the learning algorithm can interactively query a user (teacher or oracle) to label new data points with the true labels.
The motivation for active learning is a scenario in which we have a large pool of unlabelled data.
Passive learning, the standard framework in which a large quantity of labelled data is passed to the algorithm, requires significant effort in labelling the entire set of data.


![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/passive.png)

By using active learning, we can selectively leverage a system like crowd-sourcing, to ask human experts to selectively label some items in the data set, but not have to label the entirety. The algorithm iteratively selects examples based on some value, or uncertainty, metric, sends those unlabelled examples to an oracle who labels it and sends it back.


![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/active.png)


A real-world application of this is training an image classification model to distinguish between cats and dogs. There are millions of images out there of each, but not all are needed to train a good model. Similarly, this applies to things like classifying content of Youtube videos, where the data is inherently dense and there is a lot of it.


In several cases, active learning performs better than random sampling. The below graph shows a motivating toy example of active learning’s improvement over random selection. The entire set of data points (union of the sets of red triangles and green circles) is not linearly separable.


<p align="center">
  <img src = "https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/a2.png" />
</p>


The active learning framework reduces the selection of data to a problem of determining which are the most informative data points in the set? In active learning, the most informative data points are generally the ones that the model is most uncertain about. This leads to developing various metrics and definitions to measure, quantify, and compare uncertainty of examples. Active learning is motivated by the understanding that not all labelled examples are equally important. With uniform random sampling over all of the examples, the learned model doesn’t quite represent the division between classes. However, active learning selects examples near the class boundary, and is able to find a more representative classifier. Previous research has also shown that active learning offers improvement over standard random selection.

# Different Active Learning Frameworks
Active learning is considered to be a semi-supervised learning method, between unsupervised being using 0% of the learning examples and fully supervised being using 100% of the examples. By iteratively increasing the size of our labelled training set, we can achieve greater performance, near fully-supervised performance, with a fraction of the cost or time to train using all of the data.

## Pool-based Active Learning
In pool-based sampling, training examples are chosen from a large pool of unlabelled data. Selected training examples from this pool are labelled by the oracle.

## Stream-based Active Learning
In stream-based active learning, the set of all training examples is presented to the algorithm as a stream. Each example is sent individually to the algorithm for consideration. The algorithm must make an immediate decision on whether to label or not label this example. Selected training examples from this pool are labelled by the oracle, and the label is immediately received by the algorithm before the next example is shown for consideration.

# Uncertainty Measures
The decision for selecting the most informative data points is dependent on the uncertainty measure used in selection. In pool-based sampling, the active learning algorithm selects examples to add to the growing training set that are the most informative. The most informative examples are the ones that the classifier is the least certain about. The intuition behind selecting the most uncertain examples is that by obtaining the label for those particular examples, the examples with which the model has the least certainty are the most difficult examples, the most likely the ones near the class boundaries.
The learning algorithm will likely gain the most information about the class boundaries by observing the difficult examples. Explained below are four commonly used uncertainty measures.

## Largest Margin Uncertainty
![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/lm1.png)

The largest margin uncertainty is a best-versus-worst uncertainty comparison. The largest margin uncertainty (LMU) is the classification probability of the most likely class minus the classification probability of the least likely class. The intuition behind this metric is that if the probability of the most likely class is significantly greater than the probability of the least likely class, then the classifier is more certain about the example’s class membership. Likewise, if the probability of the most likely class is not much greater than the probability of the least likely class, then the classifier is less certain about the example’s class membership. The active learning algorithm will select the example with the minimum LMU value.

## Smallest Margin Uncertainty

![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/sm1.png)

The smallest margin uncertainty is a best-versus-second-best uncertainty comparison. The smallest margin uncertainty (SMU) is the classification probability of the most likely class minus the classification probability of the second most likely class. The intuition behind this metric is that if the probability of the most likely class is significantly greater than the probability of the second most likely class, then the classifier is more certain about the example’s class membership. Likewise, if the probability of the most likely class is not much greater than the probability of the second most likely class, then the classifier is less certain about the example’s class membership. The active learning algorithm will select the example with the minimum SMU value.

## Least Confidence Uncertainty

![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/lc1.png)

Least confidence uncertainty (LCU) is selecting the example for which the classifier is least certain about the selected class. LCU selection only looks at the most likely class, and selects the example that has the lowest probability assigned to that class.

## Entropy Reduction

![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/ent1.png)

Entropy is the measure of the uncertainty of a random variable. In this experiment, we use Shannon Entropy. Shannon entropy has several basic properties, including (1) uniform distributions have maximum uncertainty, (2) uncertainty is additive for independent events, and (3) adding an outcome with zero probability has no effect, and (4) events with a certain outcome have zero effect. Considering class predictions as outcomes, we can measure Shannon entropy of the predicted class probabilities.
Higher values of entropy imply more uncertainty in the probability distribution. In each active learning step of the algorithm, for every unlabelled example in the training set, we compute the entropy over the predicted class probabilities, and select the example with the highest entropy. The example with the highest entropy is the example for which the classifier is least certain about its class membership.

# Algorithm

The algorithm below is one for pool-based active learning. Stream-based active learning can be similarly written.

![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/algo1.png)

A principle bottleneck in large-scale classification tasks is the large number of training examples needed for training a classifier. Using active learning, we can reduce the number of training examples needed to teach a classifier by strategically selecting particular examples. Assigning value to examples using different uncertainty metrics allows the model to identify and select high-value examples in a smaller training set size.

You may see active learning referred to in literature as optimal experimental design. 

# Jupyter Notebook
The above notebook runs active learning of all four metrics on the Fruits360 dataset. 

The Fruits360 dataset is a dataset of fruits and vegetables by Mendeley Data. Fruits and vegetables were planted in the shaft of a low speed motor (3 rpm) and a short movie of 20 seconds was recorded. A Logitech C920 camera was used for filming the fruits. The images show the fruits rotated in different angles. 

![alt text](https://github.com/mzhao98/ActiveLearning_ImageClassification/blob/master/ims/c2.png)

We made modifications to the dataset in order to reduce the number of classes from 120 to 10. We selected 10 classes: Apple, Tomato, Potato, Banana, Pear, Pineapple, Pepper, Strawberry, Onion, and Plum. We combined specific strains of fruit like Crimson Snow Apple and Red Delicious Apple all into the Apple category, for each fruit category. Our final dataset is in a folder named categories_fruits_big. The current uploaded categories_fruits_big dataset is a VERY truncated version of the actual dataset. 

A similar version of the dataset can be constructed by following the instructions above. For access to our created categories_fruits_big dataset, please raise an issue.


