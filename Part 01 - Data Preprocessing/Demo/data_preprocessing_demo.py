# import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# read dataSet and create dataFrame with name data
data = pd.read_csv('Data.csv')

# separate data-features (X = independent variables) and data-targets(labels) (y = dependent variables) from the dataFrame
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# check the separated variables
print(X)
print(y)

# full fit a missing data in data-features
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# check the transformed data in data-features
print(X)

# encoding categorical data in data-features
features_tranform = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(features_tranform.fit_transform(X))

# check the transform (encoded) data
print(X)

# encoding categorical data in data-targets(labels)
targets_transfor = LabelEncoder()
y = targets_transfor.fit_transform(y)

# check the tranformed data in data-tagets(labels)
print(y)

# split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

# check the split sets
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# scaling the data-features
scalar = StandardScaler()
X_train[:, 3:] = scalar.fit_transform(X_train[:, 3:])
X_test[:, 3:] = scalar.transform(X_test[:, 3:])

# check scaled fetaures
print(X_train)
print(X_test)
