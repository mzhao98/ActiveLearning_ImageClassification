# ActiveLearning_ImageClassification
Active Learning for Multiclass Image Classification on Fruits360 Dataset
\usepackage[ruled,vlined]{algorithm2e}
\usepackage{amsmath}
\usepackage{cvpr}
\usepackage{times}
\usepackage{epsfig}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}


\section{Introduction}
Active learning is a machine learning framework in which the learning algorithm can interactively query a user (teacher or oracle) to label new data points with the true labels. The motivation for active learning is a scenario in which we have a large pool of unlabelled data. An example of this is training an image classification model to distinguish cats and dogs. There are millions of images out there of each, but not all are needed to train a good model. Similarly, this applies to things like classifying content of Youtube videos, where the data is inherently dense and there is a lot of it. Passive learning, the standard framework in which a large quantity of labelled data is passed to the algorithm requires significant effort in labelling the entire set of data. By using active learning, we can selectively leverage a system like crowd-sourcing, to ask human experts to selectively label some items in the data set, but not have to label the entirety. The algorithm iteratively selects examples based on some value, or uncertainty, metric, sends those unlabelled examples to an oracle who labels it and sends it back. 
\\\\In several cases, active learning performs better than random sampling. The below graph shows a motivating toy example of active learning's improvement over random selection. The entire set of data points (union of the sets of red triangles and green circles) is not linearly separable. 

\begin{figure}[!h]
	\centering
	\includegraphics[trim={0pt 0pt 0pt 0pt}, width=.35\textwidth]{a2.png}
	\caption{Illustrative Example. Active learning enables better than random selection in classification problems.}
	\label{fig:statement:model}
\end{figure}

The active learning framework reduces the selection of data to a problem of determining which are the most informative data points in the set? In active learning, the most informative data points are generally the ones that the model is most uncertain about. This leads to developing various metrics and definitions to measure, quantify, and compare uncertainty of examples. Active learning is motivated by the understanding that not all labelled examples are equally important. With uniform random sampling over all of the examples, the learned model doesn't quite represent the division between classes. However, active learning selects examples near the class boundary, and is able to find a more representative classifier. Previous work has also shown that active learning offers improvement over standard random selection.
\\\\The primary contributions of this paper are (1) an active learning algorithm that can perform multiclass classification problems, (2) a formal comparison of four uncertainty measures, and (3) an investigation of active learning on simple versus complex classification tasks.

\section{Related Work}
Training machine learning models can be expensive in the the number of training examples needed (sources here). Active learning aims to reduce the number of examples needed by choosing the most informative samples to work with (sources here). This methodology has been shown to work on *task1, task2, task3 (sources here). 

Specifically, we have seen that in image classification several techniques have been shown to perform effectively (sources here). Delve deeper into methods. 

Although several different methods have been shown to reduce the number of training samples needed, there hasn't been a formal comparison done on these methods. We aim to compare these and explain their differences. 

\section{Active Learning}
Active learning is considered to be a semi-supervised learning method, between unsupervised being using 0\% of the learning examples and fully supervised being using 100\% of the examples. By iteratively increasing the size of our labelled training set, we can achieve greater performance, near fully-supervised performance, with a fraction of the cost or time to train using all of the data.
\subsection{Pool-based Active Learning}
In pool-based sampling, training examples are chosen from a large pool or unlabelled data. Selected training examples from this pool are labelled by the oracle. For this project, we perform pool-based sampling only in our active learning framework.
\subsection{Stream-based Active Learning}
In stream-based active learning, the set of all training examples is presented to the algorithm as a stream. Each example is sent individually to the algorithm for consideration. The algorithm must make an immediate decision on whether to label or not label this example. Selected training examples from this pool are labelled by the oracle, and the label is immediately received by the algorithm before the next example is shown for consideration.

\section{Uncertainty Measures}
The decision for selecting the most informative data points is dependent on the uncertainty measure used in selection. In pool-based sampling, the active learning algorithm selects examples to add to the growing training set that are the most informative. The most informative examples are the ones that the classifier is the least certain about. The intuition behind selecting the most uncertain examples is that by obtaining the label for those particular examples, the examples with which the model has the least certainty are the most difficult examples, the most likely the ones near the class boundaries.
The learning algorithm will likely gain the most information about the class boundaries by observing the difficult examples. In this paper, we compared four uncertainty measures.

\subsection{Largest Margin Uncertainty}
$$\phi_{LM}(x) = P_{\theta}(y_1^*|x) - P_{\theta}(y_{min}^*|x)$$
The largest margin uncertainty is a best-versus-worst uncertainty comparison. The largest margin uncertainty (LMU) is the classification probability of the most likely class minus the classification probability of the least likely class. The intuition behind this metric is that if the probability of the most likely class is significantly greater than the probability of the least likely class, then the classifier is more certain about the example's class membership. Likewise, if the probability of the most likely class is not much greater than the probability of the least likely class, then the classifier is less certain about the example's class membership. The active learning algorithm will select the example with the minimum LMU value.

\subsection{Smallest Margin Uncertainty}
$$\phi_{SM}(x) = P_{\theta}(y_1^*|x) - P_{\theta}(y_2^*|x)$$
The smallest margin uncertainty is a best-versus-second-best uncertainty comparison. The smallest margin uncertainty (SMU) is the classification probability of the most likely class minus the classification probability of the second most likely class. The intuition behind this metric is that if the probability of the most likely class is significantly greater than the probability of the second most likely class, then the classifier is more certain about the example's class membership. Likewise, if the probability of the most likely class is not much greater than the probability of the second most likely class, then the classifier is less certain about the example's class membership. The active learning algorithm will select the example with the minimum SMU value.

\subsection{Least Confidence Uncertainty}
$$\phi_{LC}(x) = 1-P_{\theta}(y_1^*|x)$$
Least confidence uncertainty (LCU) is selecting the example for which the classifier is least certain about the selected class. LCU selection only looks at the most likely class, and selects the example that has the lowest probability assigned to that class.

\subsection{Entropy Reduction}
$$\phi_{ENT}(x) = -\sum_y P_{\theta}(y|x) \log P_{\theta}(y|x)$$
Entropy is the measure of the uncertainty of a random variable. In this experiment, we use Shannon Entropy. Shannon entropy has several basic properties, including (1) uniform distributions have maximum uncertainty, (2) uncertainty is additive for independent events, and (3) adding an outcome with zero probability has no effect, and (4) events with a certain outcome have zero effect. Considering class predictions as outcomes, we can measure Shannon entropy of the predicted class probabilities.
\\\\Higher values of entropy imply more uncertainty in the probability distribution. In each active learning step of the algorithm, for every unlabelled example in the training set, we compute the entropy over the predicted class probabilities, and select the example with the highest entropy. The example with the highest entropy is the example for which the classifier is least certain about its class membership.

\section{Algorithm}
\begin{algorithm}[H]
\SetAlgoLined
 $\epsilon$ = training error bound\;
 Divide data into unlabelled pool $P$ and test set $S$\;
 Split training pool into batches\;
 Randomly select $k$ examples from training pool to put in initialized training set $T$\;
 \While{Training Error $> \epsilon$}{
  Train the model using $T$ \;
  Use the trained model with the test-set, get performance measures\;
  For $e \in P$, compute uncertainty for $e$\;
  Select $k$ most-informative samples based on uncertainty metric\;
  Move these $k$ examples to training set\;
    Remove these $k$ examples from pool $P$\;
 }
 \caption{Pool-based Active Learning}
\end{algorithm}
