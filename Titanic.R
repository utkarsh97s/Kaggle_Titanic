# Artificial Neural Network

# Importing the dataset
dataset = read.csv('train.csv')
dataset = dataset[c(2,3,5,6,7,8,10,12)]
dataset_test = read.csv('test.csv')
dataset_test = dataset_test[c(2,4,5,6,7,9,11)]
# Encoding the categorical variables as factors
dataset$Sex = as.numeric(factor(dataset$Sex,
                                      levels = c('male', 'female'),
                                      labels = c(1, 2)))
dataset$Embarked = as.numeric(factor(dataset$Embarked,
                                   levels = c('S', 'C','Q'),
                                   labels = c(1, 2,3)))

dataset_test$Sex = as.numeric(factor(dataset_test$Sex,
                                levels = c('male', 'female'),
                                labels = c(1, 2)))
dataset_test$Embarked = as.numeric(factor(dataset_test$Embarked,
                                     levels = c('S', 'C','Q'),
                                     labels = c(1, 2,3)))

# Taking care of Missing data
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)


# Taking care of Missing data
dataset_test$Age = ifelse(is.na(dataset_test$Age),
                     ave(dataset_test$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset_test$Age)


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(13)
split = sample.split(dataset$Survived, SplitRatio = 0.999)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)



# Fitting ANN to the Training set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Survived',
                         training_frame = as.h2o(training_set),
                         activation = 'Rectifier',
                         hidden = c(5,5,5,5),
                         epochs = 100000,
                         train_samples_per_iteration = -2)

# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(dataset_test))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(dataset_test[, 1], y_pred)

# h2o.shutdown()