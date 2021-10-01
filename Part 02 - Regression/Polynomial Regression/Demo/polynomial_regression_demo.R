# read data
data = read.csv('Position_Salaries.csv')
data = data[2:3]

# create linear regression model and train it
lin_regr_model = lm(formula = Salary ~ Level,
                data = data)

# create polynomial regression model and train it
data$Level2 = data$Level^2
data$Level3 = data$Level^3
data$Level4 = data$Level^4
data$Level5 = data$Level^5
poly_regr_model = lm(formula = Salary ~ .,
                      data = data)

# visualize the Linear Regression
library(ggplot2)
ggplot() + 
  geom_point(aes(x = data$Level, y = data$Salary),
             colour = 'red') +
  geom_line(aes(x = data$Level, y = predict(lin_regr_model, newdata = data)),
            colour = 'blue') +
  ggtitle('Truth or Bluff in Linear Regression') +
  xlab('Position Level') + 
  ylab('Salary')

# visualize the Polynomial Regression
ggplot() + 
  geom_point(aes(x = data$Level, y = data$Salary),
             colour = 'red') +
  geom_line(aes(x = data$Level, y = predict(poly_regr_model, newdata = data)),
            colour = 'blue') +
  ggtitle('Truth or Bluff in Polynomial Regression') +
  xlab('Position Level') + 
  ylab('Salary')

# predict a new result with Linear Regression
lin_y_pred = predict(lin_regr_model, data.frame(Level = 6.5))

# predict a new result with Polynomial Regression
poly_y_pred = predict(poly_regr_model, data.frame(Level = 6.5,
                                                  Level2 = 6.5^2,
                                                  Level3 = 6.5^3,
                                                  Level4 = 6.5^4,
                                                  Level5 = 6.5^5))
