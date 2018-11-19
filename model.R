require(plyr)
require(dplyr)
require(data.table)
require(ggplot2)
require(caret)
colnames(train2)

setwd("/home/ronny/Documents/competation/Hackthons")
train=read.csv(file="train.csv")
test=read.csv(file = "test.csv")
glimpse(train)
head(train)
train=dplyr::select(train,-c(DateTime,user_id))
dim(train)
sum(is.na(train$product_category_2))/length(train$product_category_2)
summary(train)
sum(is.na(train$city_development_index))/length(train$city_development_index)

#######3drop product category 2
colnames(train)
glimpse(train)

train[train==""]=NA
train[train==" "]=NA
dim(train)

colnames(train)
train=na.omit(train)
sum(is.na(train))
dim(train)
table(train$product)
train$is_click[train$is_click=="0"]="no_ click"
train$is_click[train$is_click=="1"]="click"
train$is_click=factor(train$is_click,levels = c("click","no click"),
                      labels = c("click","no_click"))
table(train$gender)
levels(train$is_click)
dim(train)
glimpse(train)
train$campaign_id=as.factor(train$campaign_id)
train$webpage_id=as.factor(train$webpage_id)
train$product_category_1=as.factor(train$product_category_1)
train$product_category_2=as.factor(train$product_category_2)
train$user_group_id=as.factor(train$user_group_id)
train$age_level=as.factor(train$age_level)
train$user_depth=as.factor(train$user_depth)
train$city_development_index=as.factor(train$city_development_index)
train$var_1=as.factor(train$var_1)
glimpse(train)

clicks=train$is_click
require(dummies)
enc_dat=dummy.data.frame(train,names = c("product","campaign_id","webpage_id","product_category_1","product_category_2","user_group_id","gender","age_level","user_depth","city_development_index","var_1"),sep="_")
dim(enc_dat)
glimpse(enc_dat)
write.csv(enc_dat,file="enc_train.csv")
enc=read.csv(file="enc_train.csv")

data=enc[1:10000,-1]
summary(data)
write.csv(data,file="clean2.csv")


##########3models
index=createDataPartition(y=data$is_click, p=0.70,list=F)
train2=data[index,]
test2=data[-index,]
input_x=dplyr::select(train2,-c(is_click,session_id))
input_y=train2$is_click
session_in=train2$session_id
levels(data$is_click)
table(data$is_click)

test_x=dplyr::select(test2,-c(is_click,session_id))
test_y=test2$is_click
session_out=test2$session_id
length(session_out)
glimpse(train2)

tune_control=caret::trainControl(method = "cv",
                                 number = 5,
                                 classProbs = T,
                                 verboseIter = T)
dim(input_x)

xgb_tuned <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  method = "xgbTree",
  verbose = TRUE,
  tuneLength=3
)

xgb_pred=predict(xgb_tuned,test_x)
confusionMatrix(xgb_pred,test_y)
??prediction

require(ROCR)
testy=as.numeric(test_y)


xgb_pred2=as.numeric(xgb_pred)
class(test_y)
xgb_preds=prediction(xgb_pred2,testy)
roc.perf = performance(xgb_preds, measure = "tpr", x.measure = "fpr")
auc.train = performance(xgb_preds,measure = "auc")
auc.train = auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .2, y = .8,paste("AUC = ", round(auc.train[[1]],3), sep = ""))

session_ID=session_out
submission1=data.frame(session_ID,xgb_pred)

head(submission1)

write.csv(submission1,file="sub1_xgboost_pred.csv")
############
rf_tuned <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  method = "rf",
  verbose = TRUE,
  tuneLength=5
)



rf_pred=predict(rf_tuned,test_x)
confusionMatrix(rf_pred,test_y)



require(MASS)
model_lda=lda(is_click~.,data=train2)

gbm_tuned <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  method = "gbm",
  verbose = TRUE,
  tuneLength=5
)

gbm_pred=predict(gbm_tuned,test_x)
confusionMatrix(gbm_pred,test_y)

glm_tuned <- caret::train(
  x = input_x,
  y = input_y,
  trControl = tune_control,
  method = "xgbLinear",
  tuneLength=5
)




