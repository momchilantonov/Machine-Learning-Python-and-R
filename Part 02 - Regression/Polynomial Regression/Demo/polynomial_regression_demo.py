# import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# read the data
# use pandas dataFrame or Series
# data = pd.read_csv('Position_Salaries.csv', usecols=['Level', 'Salary'])
# data.shape
# use the data to create np.aaray
data = pd.read_csv('Position_Salaries.csv')

# split data-features (X = independent variables) and data-targets(labels) (y = dependent variables) from the dataFrame
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# check the separated variables
print(X)
print(y)

# create linear regression model and train it
lin_regr_model = LinearRegression()
lin_regr_model.fit(X, y)

# create polynomial regression model on a power of 2 and train it
poly_features_2 = PolynomialFeatures(degree=2)
X_poly_2 = poly_features_2.fit_transform(X)
poly_regr_model_2 = LinearRegression()
poly_regr_model_2.fit(X_poly_2, y)

# create polynomial regression model on a power of 3 and train it
poly_features_3 = PolynomialFeatures(degree=3)
X_poly_3 = poly_features_3.fit_transform(X)
poly_regr_model_3 = LinearRegression()
poly_regr_model_3.fit(X_poly_3, y)

# create polynomial regression model on a power of 4 and train it
poly_features_4 = PolynomialFeatures(degree=4)
X_poly_4 = poly_features_4.fit_transform(X)
poly_regr_model_4 = LinearRegression()
poly_regr_model_4.fit(X_poly_4, y)

# create polynomial regression model on a power of 5 and train it
poly_features_5 = PolynomialFeatures(degree=5)
X_poly_5 = poly_features_5.fit_transform(X)
poly_regr_model_5 = LinearRegression()
poly_regr_model_5.fit(X_poly_5, y)

# visualize the Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_regr_model.predict(X), color='blue')
plt.title('Truth or Bluff in Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualize the Polynomial Regression ^ 2
plt.scatter(X, y, color='red')
plt.plot(X, poly_regr_model_2.predict(X_poly_2), color='blue')
plt.title('Truth or Bluff in Polynomial Regression ^ 2')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualize the Polynomial Regression ^ 3
plt.scatter(X, y, color='red')
plt.plot(X, poly_regr_model_3.predict(X_poly_3), color='blue')
plt.title('Truth or Bluff in Polynomial Regression ^ 3')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualize the Polynomial Regression ^ 4
plt.scatter(X, y, color='red')
plt.plot(X, poly_regr_model_4.predict(X_poly_4), color='blue')
plt.title('Truth or Bluff in Polynomial Regression ^ 4')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualize the Polynomial Regression ^ 5
plt.scatter(X, y, color='red')
plt.plot(X, poly_regr_model_5.predict(X_poly_5), color='blue')
plt.title('Truth or Bluff in Polynomial Regression ^ 5')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# visualize the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, poly_regr_model_4.predict(
    poly_features_4.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff in Polynomial Regression ^ 4 (HR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# predict a new result with Linear Regression
lin_regr_model.predict([[6.5]])

# predict a new result with Polynomial Regression
poly_regr_model_4.predict(poly_features_4.fit_transform([[6.5]]))
