# read the data
data = read.csv('Social_Network_Ads.csv')

# split the data into the training and test sets
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.75) # choose the correct split ratio and the cols of the data
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

# feature scaling
training_set[, 1:2] = scale(training_set[, 1:2]) # choose correct cols
test_set[, 1:2] = scale(test_set[, 1:2]) # choose correct cols

# create model for train and predict the test set result
library(class)
y_pred = knn(train = training_set[, -3],
             test = test_set[, -3],
             cl = training_set[, 3],
             k = 5)

# create confusion matrix
conf_matrix = table(test_set[, 3], y_pred) # choose the col

# visualize the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[, -3],
               test = grid_set[, -3],
               cl = training_set[, 3],
               k = 5)
plot(set[, -3],
     main = 'KNN (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# visualize the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = knn(train = training_set[, -3],
               test = grid_set[, -3],
               cl = training_set[, 3],
               k = 5)
plot(set[, -3],
     main = 'KNN (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
