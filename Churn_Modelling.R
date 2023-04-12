library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)

#Q1. Import dataset
df <- read.csv('Churn_Modelling (1).csv')
df %>% glimpse()
df %>% skim()

#Q2. Remove unneeded columns

#Converting the label data type to factor
df$Exited <- df$Exited %>% 
  factor(levels = c(1,0),
         labels = c(1,0))#Target Column should always be in factor data type

#Showing the proportion of classes. Our label is underbalanced
df$Exited %>% table() %>% prop.table()

#Removing the columns that obviously will not have effect on churn. These are RowNumber, CustomerId, Surname.
df <- df[, -c(1, 2, 3)]

#No NA values
df %>% inspect_na()

#Calculating the importance for each column
iv <- df %>% 
  iv(y = 'Exited') %>% as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))

#We can see that NumOfProducts and Age are the most important variables in predicting churn
iv

#Getting the names of important features
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]]

#Creating a dataframe that includes only the important features
df.iv <- df %>% select(Exited,ivars)

#Creating woe bins
bins <- df.iv %>% woebin("Exited")

#Checking the bins for the Age variable. We can see that the probability that the customer will leave increases with increasing age
bins$Age %>% tibble() 
bins$Age %>% woebin_plot()


#Creating a new dataframe that includes only the woe values
df.woe <- df.iv %>% woebin_ply(bins)
names(df.woe) <- df.woe %>% names() %>% gsub("_woe","",.)

#Now we will remove multicolinear features
target <- 'Exited'
features <- df.woe %>% select(-Exited) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")
glm %>% summary()

#Removing linearly dependent columns
coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")

#Removing features with high VIF score
while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df.woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 
df.woe <- df.woe %>% select(target,features)

#Splitting the data into train and test
dt_list <- df.woe %>% 
  split_df("Exited", ratio = 0.8, seed = 123)

train_woe <- dt_list$train
test_woe <- dt_list$test 

#Q3. Build Churn model
h2o.init()

train_h2o <- train_woe %>%  as.h2o()
test_h2o <- test_woe %>%  as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

#Removing the features that have low effect on target variable (high p-value)
while(model@model$coefficients_table %>%
      as.data.frame() %>%
      select(names,p_value) %>%
      mutate(p_value = round(p_value,3)) %>%
      .[-1,] %>%
      arrange(desc(p_value)) %>%
      .[1,2] >= 0.05){
  model@model$coefficients_table %>%
    as.data.frame() %>%
    select(names,p_value) %>%
    mutate(p_value = round(p_value,3)) %>%
    filter(!is.nan(p_value)) %>%
    .[-1,] %>%
    arrange(desc(p_value)) %>%
    .[1,1] -> v
  features <- features[features!=v]
  
  train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
  test_h2o <- test_woe %>% select(target,features) %>% as.h2o()
  
  model <- h2o.glm(
    x = features, y = target, family = "binomial", 
    training_frame = train_h2o, validation_frame = test_h2o,
    nfolds = 10, seed = 123, remove_collinear_columns = T,
    balance_classes = T, lambda = 0, compute_p_values = T)
}


#We have removed all variables with p-value more than 0.05
model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>% arrange(desc(p_value))

#Checking the model coefficients
model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)

#Looking at the importance of each feature. Age is the most important variable
h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)

#Q4 & Q5. Compare model results for training and test sets & Evaluate and explain model results using ROC & AUC curves.

#Making predictions for test data
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

#Getting the confusion matrix. The accuracy of our model is 0.79.
conf_matrix <- confusionMatrix(pred$predict, test_h2o$Exited %>% as.vector() %>% 
                  factor(levels = c(0,1),
                         labels = c(0,1)))
conf_matrix

#We will now calculate precision and recall manually
results <- cbind(pred$predict %>% as.character() %>% as.numeric(), test_h2o$Exited %>% as.vector() %>% as.numeric()) %>% as.data.frame()
names(results) <- c('pred','actual')

FP <- results %>% filter(pred == 1 & actual == 0) %>% nrow()
TP <- results %>% filter(pred == 1 & actual == 1) %>% nrow()
TN <- results %>% filter(pred == 0 & actual == 0) %>% nrow()
FN <- results %>% filter(pred == 0 & actual == 1) %>% nrow()

accuracy <- (TP+TN)/(TP+TN+FP+FN)
precision <- (TP)/(TP+FP)
recall <- (TP)/(TP+FN)
f1 <- (2*precision*recall)/(precision+recall)

#Despite the fact that the accuracy for our model is high, precision is quite low.
tibble(accuracy=accuracy,
       precision=precision,
       recall=recall,
       f1_score=f1)

#In our case, the threshold that will maximize the f1 score would be 0.289.
model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

#Showing the ROC curve
eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")

#Checking for overfitting. Since auc and gini are higher for test data, we conclude that there is no overfitting.
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)
