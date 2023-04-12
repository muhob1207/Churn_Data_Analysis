# Import libraries & dataset ----
library(tidyverse)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(scorecard)#Used for woe bins

df <- germancredit

df %>% glimpse()
df %>% skim()
df$creditability

df$creditability <- df$creditability %>% 
  factor(levels = c('bad','good'),
         labels = c(1,0))#Target Column should always be in factor data type

df$creditability %>% table() %>% prop.table()#Shows the proportion of the classes (0,1)


# --------------------------------- Modeling ---------------------------------

# Weight Of Evidence ----

# IV (information values) 
iv <- df %>% 
  iv(y = 'creditability') %>% as_tibble() %>%
  mutate(info_value = round(info_value, 3)) %>%
  arrange(desc(info_value))
#Information values shows the importance of the variables and here we calculate iv according to the 
#target variable

#iv<0.02 not important
#0.02<iv<0.1 Weak predictive power
#0.1<iv<0.3 Medium predictive power
#0.3<iv<0.5 Strong predictive power
#iv>0.5 # Very Strong predictive power
# Exclude not important variables 
ivars <- iv %>% 
  filter(info_value>0.02) %>% 
  select(variable) %>% .[[1]] #Excluding not important variables.

df.iv <- df %>% select(creditability,ivars)#When we calculate iv we remove target column and here
#we merge it again

df.iv %>% dim()
# woe binning---- 
bins <- df.iv %>% woebin("creditability")# Creating woe bins. 
#From here we are going to work with woebins

#Rules for Woe Binning:

#Each category (bin) should have at least 5% of the observations.
#Each category (bin) should be non-zero for both non-events and events.
#The WOE should be distinct for each category. Similar groups should be aggregated.
#The WOE should be monotonic, i.e. either growing or decreasing with the groupings.
#Missing values are binned separately.

#Benefits of Woe:
# It can treat outliers. Suppose you have a continuous variable such as annual salary 
#and extreme values are more than 500 million dollars. These values would be grouped 
#to a class of (let's say 250-500 million dollars). 
#Later, instead of using the raw values, we would be using WOE scores of each classes.

# It can handle missing values as missing values can be binned separately.

# Since WOE Transformation handles categorical variable so there is no need for dummy variables.

# WoE transformation helps you to build strict linear relationship with log odds. 
#Otherwise it is not easy to accomplish linear relationship using other 
#transformation methods such as log, square-root etc. 
#In short, if you would not use WOE transformation, 
#you may have to try out several transformation methods to achieve this.


duration.in.month <- bins$duration.in.month %>% tibble() 

# converting into woe values


df.woe <- df.iv %>% woebin_ply(bins)

names(df.woe) <- df.woe %>% names() %>% gsub("_woe","",.)

# Multicollinearity ----

# coef_na
target <- 'creditability'
features <- df.woe %>% select(-creditability) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")#Binomial means for binary classification
glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]#Choosing linearly dependent columns
features <- features[!features %in% coef_na]#Removing them from the dataset

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")

# VIF (Variance Inflation Factor) 
f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df.woe, family = "binomial")

while(glm %>% vif() %>% arrange(desc(gvif)) %>% .[1,2] >= 1.5){
  afterVIF <- glm %>% vif() %>% arrange(desc(gvif)) %>% .[-1,"variable"]#Removing first variable with the highest
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df.woe, family = "binomial")
}

glm %>% vif() %>% arrange(desc(gvif)) %>% pull(variable) -> features 

df.woe <- df.woe %>% select(target,features)

dt_list <- df.woe %>% 
  split_df("creditability", ratio = 0.8, seed = 123)#This time we used split_df instead of h2o.split
#because we are going to work with woe bins

train_woe <- dt_list$train
test_woe <- dt_list$test 


# Modeling with H2O ----
h2o.init()

train_h2o <- train_woe %>%  as.h2o()
test_h2o <- test_woe %>%  as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

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

model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>% arrange(desc(p_value))

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)#Pulling the coefficients for each variables 

h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)#Visualizing the Influence of each variable for prediction


# ---------------------------- Evaluation Metrices ----------------------------

# Prediction & Confision Matrice
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)#Predicting values

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')#Finding the threshold by f1 score

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = dt_list$test$creditability %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")#Roc curve 
#auc-Area under the curve

eva$confusion_matrix$dat

# Check overfitting ----
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)#Checking Performance by gini score

model %>% 
  h2o.confusionMatrix(test_h2o) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t() %>% 
  fourfoldplot(conf.level = 0, color = c("red", "darkgreen"),
               main = paste("Accuracy = ",
                            round(sum(diag(.))/sum(.)*100,1),"%"))

model %>% h2o.performance(newdata = test_h2o) %>% h2o.recall()
