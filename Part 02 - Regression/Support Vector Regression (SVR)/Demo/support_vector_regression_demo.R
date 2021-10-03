# read data
data = read.csv('Position_Salaries.csv')
data = data[2:3]

# create SVR model and train it
# install.packages('e1071')
library(e1071)
svr_regr_model = svm(formula = Salary ~ .,
                     data = data,
                     type = 'eps-regression')

# predict a new result
y_pred = predict(svr_regr_model, data.frame(Level = 6.5))

# visualize the SVR
library(ggplot2)
ggplot() +
  geom_point(aes(x = data$Level, y = data$Salary),
             colour = 'red') +
  geom_line(aes(x = data$Level, y = predict(svr_regr_model, newdata = data)),
            colour = 'blue') +
  ggtitle('Truth or Bluff in SVR') +
  xlab('Position Level') +
  ylab('Salary')
