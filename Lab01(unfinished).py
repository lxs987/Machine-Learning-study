# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 01:21:56 2020

@author: lxs_9
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

exist_wrong_value = False
is_plot = False
set_K_best = False
k_best_value = 10
# Variable determining whether to use PCA
pca_parameter = None  # integer value or None

df = pd.read_csv('C:\\Users\lxs_9\Downloads\heart.csv')


def select_k_best(df, n):
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    # Num of Top feature select...
    bestFeatures = SelectKBest(score_func=chi2, k=n)
    fit = bestFeatures.fit(X, Y)
    dfColumns = pd.DataFrame(X.columns)
    dfscores = pd.DataFrame(fit.scores_)
    featureScores = pd.concat([dfColumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']
    print(featureScores.nlargest(n, 'Score'))
    return featureScores.nlargest(n, 'Score')


def ScalingData(data, scaler):
    df_scaled = scaler.fit_transform(data)
    return df_scaled


# function of LabelEncoding
def columnEncoding(col, df):
    for c in col:
        # get All distinguish Value of column
        c_unique = df[c].unique()
        print(c_unique)
        encoder = LabelEncoder()
        encoder.fit(df[c])
        df[c] = encoder.transform(df[c])
        result_unique = df[c].unique()
        print("%s is replaced %s" % (c_unique, result_unique))  # print replaced label compared origin label
    return df


# ─────────────────────────────────────────────────────────────
#                Process & Actual Action
# ─────────────────────────────────────────────────────────────

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score


# Preprocessing
# Part that determines how to handle na value of dataframe
def handleNa(col, df, work="mean"):
    # fill na value for 'mean'
    if work == "mean":
        for c in col:
            mean = df[c].mean()
            df[c] = df[c].fillna(mean)
    # fill na value for 'median'
    elif work == "median":
        for c in col:
            median = df[c].median()
            df[c] = df[c].fillna(median)
    # fill na value for 'mode'
    elif work == "mode":
        for c in col:
            mode = df[c].mode()[0]
            df[c] = df[c].fillna(mode)
    # drop row which contains na value
    elif work == "drop":
        df = df.dropna(subset=col)
    return df


# ─────────────────────────────────────────────────────────────
#            DataFrame Scaling with Function
# ─────────────────────────────────────────────────────────────

r_scaler = RobustScaler()
s_scaler = StandardScaler()
m_scaler = MinMaxScaler()

s_scaled_df = ScalingData(df, s_scaler)
r_scaled_df = ScalingData(df, r_scaler)
m_scaled_df = ScalingData(df, m_scaler)

# ─────────────────────────────────────────────────────────────
#   데이터가 다 수치화되어있고 missing 데이터가 없는데 어떻게 encoding이랑 cleansing을 하죠
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
#         Building Decision Tree Model
# ─────────────────────────────────────────────────────────────
print('')
print('#─────────────────────────────────────────────────────────────')
print('#             1. Building Decision Tree Model')
print('#─────────────────────────────────────────────────────────────')
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
                'thal']
X = df[feature_cols]  # Features
y = df.target  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("DecisionTree Accuracy:", metrics.accuracy_score(y_test, y_pred))

print('k-fold cross validation (k = 10) of Decision-Tree Model')

kfold = KFold(n_splits=10)  # KFold 객체 생성

for n in range(2, 12):
    kfold = KFold(n_splits=n)

    scores = cross_val_score(clf, X, y, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(n, scores))

# ─────────────────────────────────────────────────────────────
#         Building Linear Regression Model
# ─────────────────────────────────────────────────────────────
print('')
print('#─────────────────────────────────────────────────────────────')
print('#             2. Building Linear Regression Model')
print('#─────────────────────────────────────────────────────────────')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=10000)

# If running this code without parameter (max_iter=10000), It makes warning like under lines.

# C:\Users\lxs_9\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:762: ConvergenceWarning: lbfgs failed to converge (status=1):
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

# Increase the number of iterations (max_iter) or scale the data as shown in:
#     https://scikit-learn.org/stable/modules/preprocessing.html
# Please also refer to the documentation for alternative solver options:
#     https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#   n_iter_i = _check_optimize_result(


# fit the model with data
logreg.fit(X_train, y_train)

#
y_pred = logreg.predict(X_test)
print("Linear Regression Accuracy:", metrics.accuracy_score(y_test, y_pred))

print('k-fold cross validation (k = 10) of Linear-Regression Model')

kfold = KFold(n_splits=10)  # KFold 객체 생성

for n in [3, 5]:
    kfold = KFold(n_splits=n)

    scores = cross_val_score(logreg, X, y, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(n, scores))

# ─────────────────────────────────────────────────────────────
#         Model Evaluation using Confusion Matrix
# ─────────────────────────────────────────────────────────────
print('')
print('#─────────────────────────────────────────────────────────────')
print('#             + Model Evaluation using Confusion Matrix')
print('#─────────────────────────────────────────────────────────────')
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('Model Evaluation using Confusion Matrix')
print(cnf_matrix)

# ─────────────────────────────────────────────────────────────
#         Visualizing Confusion Matrix using Heatmap
# ─────────────────────────────────────────────────────────────
print('')
print('#─────────────────────────────────────────────────────────────')
print('#             + Visualizing Confusion Matrix using Heatmap')
print('#─────────────────────────────────────────────────────────────')
class_names = [0, 1]  # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# ─────────────────────────────────────────────────────────────
#                  ROC Curve
# ─────────────────────────────────────────────────────────────
print('')
print('#─────────────────────────────────────────────────────────────')
print('#             + ROC Curve')
print('#─────────────────────────────────────────────────────────────')
y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

# ─────────────────────────────────────────────────────────────
#       SVM (Support Vector Machines with Scikit-Learn)
#       https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python
# ─────────────────────────────────────────────────────────────
print('')

print('#─────────────────────────────────────────────────────────────')
print('#             3. SVM (Support Vector Machines with Scikit-Learn)')
print('#─────────────────────────────────────────────────────────────')
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)  # 70% training and 30% test

# Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("SVM Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("SVM Precision:", metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("SVM Recall:", metrics.recall_score(y_test, y_pred))

print('\nk-fold cross validation (k = 10) of SVM Model')

kfold = KFold(n_splits=10)  # KFold 객체 생성

for n in [3, 5]:
    kfold = KFold(n_splits=n)

    scores = cross_val_score(clf, X, y, cv=kfold)

    print('n_splits={}, cross validation score: {}'.format(n, scores))

# ─────────────────────────────────────────────────────────────
#       Binary Classification
#       https://machinelearningmastery.com/types-of-classification-in-machine-learning/
# ─────────────────────────────────────────────────────────────

print('')
print('#─────────────────────────────────────────────────────────────')
print('#             Binary Classification')
print('#─────────────────────────────────────────────────────────────')

from numpy import where
from collections import Counter
from sklearn.datasets import make_blobs
from matplotlib import pyplot

# define dataset
X, y = make_blobs(n_samples=1000, centers=2, random_state=1)
# summarize dataset shape
print(X.shape, y.shape)
# summarize observations by class label
counter = Counter(y)
print(counter)
# summarize first few examples
for i in range(10):
    print(X[i], y[i])
# plot the dataset and color the by class label
for label, _ in counter.items():
    row_ix = where(y == label)[0]
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()