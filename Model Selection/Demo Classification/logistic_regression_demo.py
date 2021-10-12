# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# read the data
data = pd.read_csv('Data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# create the model and train it
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# create the confusion matrix and check the score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
