# imports
from matplotlib import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# read data
data = pd.read_csv('Position_Salaries.csv')

# separate the data
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# check the saparated data
# print(X)
# print(y)

# transform (reshape) y, couse we need 2D in the scaling method
y = y.reshape(len(y), 1)

# check the transformed y
# print(y)

# scaling the data
scalar_X = StandardScaler()
scalar_y = StandardScaler()
X_scaled = scalar_X.fit_transform(X)
y_scaled = scalar_y.fit_transform(y)

# check the scaled data
# print(X)
# print(y)

# train the model
svr_regr_model = SVR(kernel='rbf')

# return y to the original shape, couse the fit method work with vector
svr_regr_model.fit(X_scaled, np.ravel(y_scaled))

# return the values to original scale and predict a new result
scalar_y.inverse_transform(
    [svr_regr_model.predict(scalar_X.transform([[6.5]]))])  # need to be careful about the shape of the ndarray, it must be 2D (should pass [])

# visualize the result
X_plot_scatter = scalar_X.inverse_transform(X_scaled)
y_plot = scalar_y.inverse_transform([svr_regr_model.predict(X_scaled)]).T
y_scatter = scalar_y.inverse_transform(y_scaled)
plt.scatter(X_plot_scatter, y_scatter, color='red')
plt.plot(X_plot_scatter, y_plot, color='blue')
plt.title('Truth or Bluff in SVR')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
