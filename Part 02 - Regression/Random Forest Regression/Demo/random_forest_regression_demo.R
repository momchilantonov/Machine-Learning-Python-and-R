# read the data
data = read.csv('Position_Salaries.csv')
data = data[2:3]

# create model for train
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(X = data[1],
                         y = data$Salary,
                         ntree = 500)

# predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# visualize (HR)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(data$Level), max(data$Level), 0.01)
ggplot() +
  geom_point(aes(x = data$Level, y = data$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff in Random Forest Regression') +
  xlab('Level') +
  ylab('Salary')
