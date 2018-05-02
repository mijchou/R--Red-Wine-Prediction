## Trees prediction using caret

# setup

intall.packages('rpart')
install.packages('caret')
library(rpart)
library(caret)

red.df <- read.csv("red.txt", sep=";")

# Check Data

str(red.df)
any(is.na(red.df)) # checking missing values

# Splitting

index <- sample(nrow(red.df)*0.1) # random index for test set

train <- red.df[-index, ]
test <- red.df[index, ]

# Modelling

model.full <- quality ~ .
grid.rt <- expand.grid(.cp = seq(0.001, 0.1, by = 0.001))
trControl <- trainControl(method = "cv", number = 10) # 10-fold Cross Validation

rtCV <- train(model.full, data = train, # model training
              method = "rpart",
              tuneGrid = grid.rt,
              trControl = trControl)

rtCV

plot(rtCV)

# Prediction and evaluation

pred <- predict(rtCV, newdata = test)
MSE <- mean((pred - test$quality)^2)

MSE

