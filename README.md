Download Link: https://assignmentchef.com/product/solved-comp9517-lab4-pattern-recognition
<br>
<h1>Pattern Recognition</h1>

To goal of this lab is to implement a K-Nearest Neighbours (KNN) classifier, a Stochastic Gradient Descent (SGD) classifier, and a Decision Tree (DT) classifier. <strong>Background information on these classifiers is provided at the end of this document</strong>.

The experiments in this lab will be based on scikit-learn’s <strong>digits</strong> data set which was designed to test classification algorithms. This data set contains 1797 low-resolution images (8 × 8 pixels) of digits ranging from 0 to 9, and the true digit value (also called the label) for each image is also given (see examples on the next page).

We will predominantly be using scikit-learn for this lab, so make sure you have downloaded it. The following scikit-learn libraries will need to be imported:

from sklearn import metrics from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split




from sklearn.neighbors import KNeighborsClassifier from sklearn.linear_model import SGDClassifier from sklearn.tree import DecisionTreeClassifier

Sample of the first 10 training images and their corresponding labels.

<strong>Task (2.5 marks):</strong> Perform image classification on the digits data set.

Develop a program to perform digit recognition. Classify the images from the digits data set using the three classifiers mentioned above and compare the classification results. The program should contain the following steps:

<h1>Set Up</h1>

<u>Step 1</u>. Import relevant packages (most listed above).

<u>Step 2</u>. Load the images using sklearn’s load_digits().

Optional: Familiarize yourself with the dataset. For example, find out how many images and labels there are, the size of each image, and display some of the images and their labels. The following code will plot the first entry (digit 0):

plt.imshow(np.reshape(digits.data[0], (8, 8)), cmap=’gray’) plt.title(‘Label: %i
’ % digits.target[0], fontsize=25)

<u>Step 3</u>. Split the images using sklearn’s train_test_split()with a test size anywhere from 20% to 30% (inclusive).

<h1>Classification</h1>

For each of the classifiers (KNeighborsClassifier, SGDClassifier, and DecisionTreeClassifier) perform the following steps:

<u>Step 4</u>. Initialize the classifier model.

<u>Step 5</u>. Fit the model to the training data.

<u>Step 6</u>. Use the trained/fitted model to evaluate the test data.

<strong> </strong>

<h1>Evaluation</h1>

<u>Step 7</u>. For each of the three classifiers, evaluate the digit classification performance by calculating the <strong>accuracy</strong>, the <strong>recall</strong>, and the <strong>confusion matrix</strong>.

Experiment with the number of neighbours used in the KNN classifier in an attempt to find the best number for this data set. You can adjust the number of neighbours with the n_neighbours parameter (the default value is 5).

Print the <strong>accuracy and recall of all three classifiers</strong> and the <strong>confusion matrix of the bestperforming classifier</strong>. Submit a screenshot for marking (see the example below for the case of just a 6-class model). Also submit your code and include a brief justification for the chosen parameter settings for KNN.




<strong>Background Information</strong>

<h1>K-Nearest Neighbours (KNN)</h1>

The KNN algorithm is very simple and very effective. The model representation for KNN is the entire training data set. Predictions are made for a new data point by searching through the entire training set for the K most similar instances (the neighbours) and summarizing the output variable for those K instances. For regression problems, this might be the mean output variable, for classification problems this might be the mode (or most common) class value. The trick is in how to determine the <strong>similarity</strong> between the data instances.

A 2-class KNN example with 3 and 6 neighbours (from <a href="https://towardsdatascience.com/knn-k-nearest-neighbors-1-a4707b24bd1d">Towards Data Science</a><a href="https://towardsdatascience.com/knn-k-nearest-neighbors-1-a4707b24bd1d">)</a>.

<strong>Similarity:</strong> To make predictions we need to calculate the similarity between any two data instances. This way we can locate the K most similar data instances in the training data set for a given member of the test data set and in turn make a prediction. For a numeric data set, we can directly use the Euclidean distance measure. This is defined as the square root of the sum of the squared differences between the two arrays of numbers.

<strong>Parameters:</strong> Refer to the scikit-learn documentation for available parameters.

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html">https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html</a>

<h1>Decision Tree (DT)</h1>

See <a href="https://en.wikipedia.org/wiki/Decision_tree_learning">https://en.wikipedia.org/wiki/Decision_tree_learning</a> for more information.

The algorithm for constructing a decision tree is as follows:

<ol>

 <li>Select a feature to place at the node (the first one is the root).</li>

 <li>Make one branch for each possible value.</li>

 <li>For each branch node, repeat Steps 1 and 2.</li>

 <li>If all instances at a node have the same classification, stop developing that part of the tree.</li>

</ol>

How to determine which feature to split on in Step 1? One way is to use measures from information theory such as <strong>Entropy</strong> or <strong>Information Gain</strong> as explained in the lecture.

<strong>Stochastic Gradient Descent (SGD)</strong>

See <a href="https://scikit-learn.org/stable/modules/sgd.html">https://scikit-learn.org/stable/modules/sgd.html</a> for more information.

<h1>Experiment with Different Classifiers</h1>

See <a href="https://scikit-learn.org/stable/modules/multiclass.html">https://scikit-learn.org/stable/modules/multiclass.html</a> for more information. There are many more models to experiment with. Here is an example of a clustering model: