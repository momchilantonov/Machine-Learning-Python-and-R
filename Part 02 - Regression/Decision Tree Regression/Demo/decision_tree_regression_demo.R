# read the data
data = read.csv('Position_Salaries.csv')
data = data[2:3]

# create model for train
# install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = data,
                  control = rpart.control(minsplit=1))

# predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# visualize (HR)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(data$Level), max(data$Level), 0.1)
ggplot() +
  geom_point(aes(x = data$Level, y = data$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')
