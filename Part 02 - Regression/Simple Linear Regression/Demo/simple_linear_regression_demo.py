# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# read data
data = pd.read_csv('Salary_Data.csv')

# separate data-features (X = independent variables) and data-targets(labels) (y = dependent variables) from the dataFrame
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# training the Simple Linear Regression model
regr_model = LinearRegression()
regr_model.fit(X_train, y_train)

# predict the Test set results
y_pred = regr_model.predict(X_test)

# visualize the Train set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regr_model.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualize the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regr_model.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# BONUS
# 1. Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regr_model.predict([[12]]))

'''
Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138967,5.

Important note: Notice that the value of the feature (12 years) was input in a double pair of square brackets. 
That's because the "predict" method always expects a 2D array as the format of its inputs. 
And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:

12→scalar 
[12]→1D array 
[[12]]→2D array
'''

# 2. Getting the final linear regression equation with the values of the coefficients
print(regr_model.coef_)
print(regr_model.intercept_)

'''
Therefore, the equation of our simple linear regression model is:

Salary=9345.94×YearsExperience+26816.19 

Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object.
Attributes in Python are different than methods and usually return a simple value or an array of values.
'''
