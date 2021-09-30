# read data
data = read.csv('50_Startups.csv')

# encoding categorical data in data-features
data$State = factor(data$State,
                      levels = c('New York', 'California', 'Florida'),
                      labels = c(1, 2, 3))

# Splitting the data set into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(data$Profit, SplitRatio = 0.8)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

# create a model and train it
# regr_model = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State)
# we can write it in a simple way
regr_model = lm(formula = Profit ~ .,
                data = training_set)

# predict the result with test set
y_pred = predict(regr_model, newdata = test_set)

# BONUS 1
# Building the optimal model using Backward Elimination
regr_model_otp = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                data = data)
summary(regr_model_otp)

# Go thru this till P value is > 0.05 for each feature

# create new model
regr_model_otp = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
                data = data)
# check results
summary(regr_model_otp)

# create new model
regr_model_otp = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
                data = data)

# check results
summary(regr_model_otp)

# create new model
regr_model_otp = lm(formula = Profit ~ R.D.Spend,
                data = data)

# check results
summary(regr_model_otp)

# BONUS 2
#Automatic Backwards Elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
