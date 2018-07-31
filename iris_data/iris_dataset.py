# Iris data set
# 3 class of flowers. Iris setosa, Iris virginica and Iris versicolor. 50 samples in each class. 

# Import package
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loading dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(10))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# 1.Univariate plots to better understand each attribute.
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# 2. Multivariate Plots
# Helpful to spot structured relationships between input variables.
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = dataset.values
X = dataset.values[:,0:4]
Y = dataset.values[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test Harness
# Use 10-fold cross validation to estimate accuracy.
# This will split my dataset into 10 parts.
# train on 9 and test on 1 and repeat for all combinations of train-test splits.
# Test options and evaluation metric
num_folds = 10
num_instances = len(X_train)
seed = 7
scoring = 'accuracy'

# Evaluating 6 different Algorithms
# Build Models 
# 1.Logistic Regression (LR)
# 2.Linear Discriminant Analysis (LDA)
# 3.K-Nearest Neighbors (KNN).
# 4.Classification and Regression Trees (CART).
# 5.Gaussian Naive Bayes (NB).
# 6.Support Vector Machines (SVM).
# Simple linear (LR and LDA) & nonlinear (KNN, CART, NB and SVM) algorithms. 
# Reset the random number seed before each run.
# This ensures that the evaluation of each algorithm is performed using exactly the same data splits.
# It ensures the results are directly comparable.

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# We now have 6 models and accuracy estimations for each. 
# Need to compare the models to each other and select the most accurate.
# plot of the model evaluation results and compare the spread and the mean accuracy of each model. 
# There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
# Running the example above, I get the above raw results:--

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make Predictions
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))