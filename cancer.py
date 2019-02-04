# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Importing the dataset

df = pd.read_csv('cancer.csv')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

print(df.head(10))
X = np.array(df.drop(['classes'], axis=1))

y = np.array(df['classes'])

# Splitting the dataset into the Training set and Test set



print('shape of the features shape: ',X.shape)
print('shape of the outcomes shape: ',y.shape)


# Make predictions on validation dataset
knn = KNeighborsClassifier(n_neighbors=5)
scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print('knn accuracy: ',scores.mean())
y_pred = cross_val_predict(knn, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)


lr=LogisticRegression()
scores=cross_val_score(lr,X,y,cv=10,scoring='accuracy')
print('lr accuracy: ',scores.mean())
y_pred = cross_val_predict(lr, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

lda=LinearDiscriminantAnalysis()
scores=cross_val_score(lda,X,y,cv=10,scoring='accuracy')
print('lda accuracy: ',scores.mean())
y_pred = cross_val_predict(lda, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

nb=GaussianNB()
scores=cross_val_score(nb,X,y,cv=10,scoring='accuracy')
print('nb accuracy: ',scores.mean())
y_pred = cross_val_predict(nb, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)


svm=SVC()
scores=cross_val_score(svm,X,y,cv=10,scoring='accuracy')
print('svm accuracy: ',scores.mean())
y_pred = cross_val_predict(svm, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)

cart=DecisionTreeClassifier()
scores=cross_val_score(cart,X,y,cv=10,scoring='accuracy')
print('cart accuracy: ',scores.mean())
y_pred = cross_val_predict(cart, X, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(conf_mat)