# read data
data = read.csv('Salary_Data.csv')

# split the data to train and test sets
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(data$Salary, SplitRatio = 2/3)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

# training the Simple Linear Regression model
regr_model = lm(formula = Salary ~ YearsExperience,
                data = training_set)

# predict the test set results
y_pred = predict(regr_model, newdata = test_set)

# visualize the train set
# install.packages('ggplot2')
library(ggplot2)
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regr_model, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Experience') + 
  ylab('Salary')

# visualize the test set
ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regr_model, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Experience') + 
  ylab('Salary')
