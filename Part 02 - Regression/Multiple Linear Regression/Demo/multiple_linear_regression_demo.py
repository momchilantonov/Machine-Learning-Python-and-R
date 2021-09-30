# import python libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# read dataSet and create dataFrame with name data
data = pd.read_csv('50_Startups.csv')

# separate data-features (X = independent variables) and data-targets(labels) (y = dependent variables) from the dataFrame
X = data.iloc[:, :4].values
y = data.iloc[:, -1].values

# check the separated variables
print(type(X))
print(type(y))

# encoding categorical data in data-features
cols_transf = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [3])], remainder='passthrough')
X = np.array(cols_transf.fit_transform(X))

# check the transform (encoded) data
print(X)

# split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# check the split sets
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# create model and train it
regr_model = LinearRegression()
regr_model.fit(X_train, y_train)

# make prediction with test set
y_pred = regr_model.predict(X_test)

# set precision option to np
np.set_printoptions(precision=3)

# concatenate the real profit with predicted one and print it to check the results
print(np.concatenate((y_pred.reshape(len(y_pred), 1),
      y_test.reshape(len(y_test), 1)), axis=1))

# BONUS 1
'''
Building the optimal model using Backward Elimination
'''

# add col from 1(ones) to X
X = np.append(arr=np.ones(shape=(50, 1)).astype(int), values=X, axis=1)

# check X with the new feature
print(X)

# create optimal matrix
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)

# create new model (Ordinary Linear Regression)
regr_model_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# check results
regr_model_OLS.summary()

'''
Go thru this till P value is > 0.05 for each feature
'''

# create optimal matrix
X_opt = X[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)

# create new model (Ordinary Linear Regression)
regr_model_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# check results
regr_model_OLS.summary()

# create optimal matrix
X_opt = X[:, [0, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)

# create new model (Ordinary Linear Regression)
regr_model_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# check results
regr_model_OLS.summary()

# create optimal matrix
X_opt = X[:, [0, 3, 5]]
X_opt = X_opt.astype(np.float64)

# create new model (Ordinary Linear Regression)
regr_model_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# check results
regr_model_OLS.summary()

# create optimal matrix
X_opt = X[:, [0, 3]]
X_opt = X_opt.astype(np.float64)

# create new model (Ordinary Linear Regression)
regr_model_OLS = sm.OLS(endog=y, exog=X_opt).fit()

# check results
regr_model_OLS.summary()

# BONUS 2
'''
Making a single prediction (for example the profit of a startup with 
R&D Spend = 160000, Administration Spend = 130000, 
Marketing Spend = 300000 and State = 'California')
'''

print(regr_model.predict([[0, 0, 160000, 130000, 300000]]))

'''
Therefore, our model predicts that the profit of a Californian startup which 
spent 160000 in R&D, 130000 in Administration and 300000 in Marketing is $ 181566,92.

Important note 1: Notice that the values of the features were all input in a double pair of square brackets. 
That's because the "predict" method always expects a 2D array as the format of its inputs. 
And putting our values into a double pair of square brackets makes the input exactly a 2D array. Simply put:

0,0,160000,130000,300000→scalars 
[0,0,160000,130000,300000]→1D array 
[[0,0,160000,130000,300000]]→2D array 

Important note 2: Notice also that the "California" state was not input as a string in the last column
but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, 
and as we see in the second row of the matrix of features X, "California" was encoded as "0, 0". 
And be careful to include these values in the first three columns, not the last three ones, 
because the dummy variables are always created in the first columns.
'''

# BONUS 3
'''
Getting the final linear regression equation with the values of the coefficients
'''

print(regr_model.coef_)
print(regr_model.intercept_)

'''
Therefore, the equation of our multiple linear regression model is:

Profit=86.6×Dummy State 1−873×Dummy State 2+786×Dummy State 3+0.773×R&D Spend+0.0329×Administration+0.0366×Marketing Spend+42467.53

Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. 
Attributes in Python are different than methods and usually return a simple value or an array of values.
'''
