# R--Red-Wine-Prediction

This repository contains different predictive methods of the red wine ratings (along with my own explanations!) based on various features. Data can be found as *red.txt* under the main directory.

Content:

1. **Regression Tree** (packages: rpart) <br/>
2. **Pruned Regression Tree** <br/>
3. **KNN** (packages: class) </br>
4. **Neural Networks**

With each model we will tune the parameters with the package **caret** (Classification And REgression Training).

## Setup

``` r
library(rpart)

library(e1071)
library(class)
library(VGAM)
library(xgboost)
```

## Data Checking

Overview the data with str() and check that no missing values are present.

``` r
str(red.df)
any(is.na(red.df)) # checking missing values
```

## Training/Test Sets Division

Split the dataset into 90% & 10% for training & test sets.

``` r
index <- sample(nrow(red.df)*0.1) # random index for test set

train <- red.df[-index, ]
test <- red.df[index, ]
```

## Modelling

The train() function from the caret package trains the model with given arguments. According to the method used, specific tuning parameters will be required to tune the model. Here we have rpart requiring cp (Complexity Parameter) as its only parameter. A grid of cp can be fed to the argument tuneGrid for search of best result (E.g. Choosing the value of cp giving the lowest RMSE.) trControl specifies the type of resampling. <br/>

Several models will be built and compared at the end of the repository.

Decision Tree (rpart - regressive partitioning)
===============================================

Tree-based methods are simiple and useful for interpretation. It can be applied to both regression and classification problems. Extended methods such as baggin, random forest an boosting are built upon the basic decision trees.

### Parameters Tuning

Grid of tuning paramters

``` r
grid.rt <- expand.grid(.cp = seq(0.001, 0.1, by = 0.001))
head(grid.rt)
```

    ##       .cp
    ## 1   0.001
    ## 2   0.002
    ## 3   0.003
    ## 4   0.004
    ## 5   0.005

``` r
trControl <- trainControl(method = "cv", number = 10) # 10-fold Cross Validation

rtCV <- train(quality ~ ., data = train, # model training
              method = "rpart",
              tuneGrid = grid.rt,
              trControl = trControl)
```

### Model Checking

Checking the model rtCV and its plot, we see that the lowest RMSE happens at cp = 0.005.

``` r
rtCV
```

    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.003.

``` r
plot(rtCV)
```

![](/Decision_Tree/Decision_Tree_with_caret_files/figure-markdown_github/unnamed-chunk-5-1.png)

### Prediction and evaluation

Final step: Making prediction with the test set on-hold. Check out the MSE (mean squared error)--the mean of squared distance between each predicted and the original value.

``` r
pred <- predict(rtCV, newdata = test)
head(pred) # prediction
```

    ##      105       43       55       50       78       90 
    ## 5.093750 5.181818 5.411255 5.102564 5.365854 5.411255

``` r
head(test$quality) # original
```

    ## [1] 5 6 6 5 6 5

To find out the Mean Squared Error (MSE) of the prediction:

``` r
MSE <- mean((pred - test$quality)^2)
MSE
```

    ## [1] 0.4174422

rpart MSE = 0.42


