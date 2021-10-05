# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# read the data and create a dataFrame
data = pd.read_csv('Position_Salaries.csv')  # select file (path)

# separate the data to features and labels
X = data.iloc[:, 1:-1].values  # features - select cols nad rows range
y = data.iloc[:, -1].values  # labels (targets) - select cols nad rows range
# check the separated data
print(f'Features[X]:\n{X}\n{20 * "-"}')
print(f'Labels[y]:\n{y}\n{20 * "-"}')

# create decision tree regression model and train it
rf_regr_model = RandomForestRegressor(n_estimators=10, random_state=0)
rf_regr_model.fit(X, y)

# predict a new value
rf_regr_model.predict([[6.5]])

# visualize the random forest
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, rf_regr_model.predict(X_grid), color='blue')
plt.title('Truth or Bluff in Random Forest Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
