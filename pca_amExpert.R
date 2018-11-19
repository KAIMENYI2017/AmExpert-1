require(plyr)
require(dplyr)
require(data.table)
require(ggplot2)
require(caret)
setwd("/home/ronny/Documents/competation/Hackthons")
data=read.csv(file="enc_train.csv")
dat=as.matrix((data))
table(data$is_click)


#matrix for to be used in calculating principal components
matrix=as.matrix(select(data,-c(session_ID,is_click)))
matrix.prc = prcomp(matrix)
par(mfrow = c(1, 1))
###variability of each component
pr.var = matrix.prc$sdev ^ 2

##variance explained 
pve =pr.var/sum(pr.var)
##
round(pr.var, 2)
round(pve, 2)
##cumulative variance
round(cumsum(pve), 2)

# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")
par(mfrow=c(1,2))
# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")

plot(matrix.prc$x[, c(1, 2)], col = (is_click)) 
legend(x="topleft", pch=1, col = c("red", "black"), legend = c("Click", "No click"))

clicks=is_click #copy object
is_click=as.numeric(data$is_click)## as numeric type
#modeling 

matrix.pcs=matrix.prc$x[,1:18] ##select first 18 principal components as suggested by cumulative variance(number of components that explain >85% variance)
head(matrix.pcs)
class(matrix.pcs)

matrix.pcs=as.data.frame(matrix.pcs)
matrix.pcs$is_click=data$is_click
matrix.pcs$session_ID=data$session_id


#write principal comp in csv file
write.csv(matrix.pcs,file="principalcomponents.csv")
head(matrix.pcs)

###split the data into train and test
index=createDataPartition(matrix.pcs$is_click,p=0.70,list=F)
train=matrix.pcs[index,]
test=matrix.pcs[-index,]
train2=dplyr::select(train,-c(session_ID))
test2=dplyr::select(test,-c(session_ID))
y=as.factor(test2$is_click)
predicted=as.factor(predicted)
head(predicted)
class(y)
dim(test)
library(MASS)

###LDA model
model_lda=lda(is_click~.,data=train2)
predicted=predict(model_lda,newdata = test2)
# Evaluate the model

library(ROCR)
confusionMatrix(predicted$class,test2$is_click)
posteriors=as.data.frame(predicted$posterior)
par(mfrow=c(1,1))
pred = prediction(posteriors[,2], test2$is_click)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train = performance(pred, measure = "auc")
auc.train = auc.train@y.values
plot(roc.perf,main="Area Under the Curve")
text(x = .2, y = .8,paste("AUC = ", round(auc.train[[1]],3), sep = ""))

head(submission2)
submission2=data.frame(test2$session_ID,predicted$class)

write.csv(submission2,file="submission_2.csv")
##########random forest
forest=caret::train(is_click~.,data=train2,method='rf',trControl = tune_control,verbose = TRUE, tuneLength=3)
pred_forest=predict(forest,newdata = test2)

confusionMatrix(pred_forest,test2$is_click)
pred_forest2=as.numeric(pred_forest)
testy=as.numeric(test2$is_click)

pre4=prediction(pred_forest2,testy)
roc.perf = performance(pre4, measure = "tpr", x.measure = "fpr")
auc.train = performance(pre4,measure = "auc")
auc.train = auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .2, y = .8,paste("AUC = ", round(auc.train[[1]],3), sep = ""))

xgb_pca=caret::train(is_click~.,data=train2,method='xgbTree',trControl = tune_control,verbose = TRUE, tuneLength=3)
xgb_pc_pred=predict(xgb_pca,newdata = test2)
confusionMatrix(xgb_pc_pred,test2$is_click)

xgb_lin_pca=caret::train(is_click~.,data=train2,method='xgbLinear',trControl = tune_control,verbose = TRUE, tuneLength=3)
pca_lin_pred=predict(xgb_lin_pca,newdata = test2)
confusionMatrix(pca_lin_pred,test2$is_click)
testy=as.numeric(test2$is_click)
pca_lin_pred=as.numeric(pca_lin_pred)


pre3=prediction(pca_lin_pred,testy)
roc.perf = performance(pre3, measure = "tpr", x.measure = "fpr")
auc.train = performance(pre3,measure = "auc")
auc.train = auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .2, y = .8,paste("AUC = ", round(auc.train[[1]],3), sep = ""))

gbm_pca=caret::train(is_click~.,data=train2,method='gbm',trControl = tune_control,verbose = TRUE, tuneLength=3)
gbm_pred_pca=predict(gbm_pca,newdata = test2)
confusionMatrix(gbm_pred_pca,test2$is_click)


y_test=as.numeric(test2$is_click)

gbm_pred_pca2=as.numeric(gbm_pred_pca)
class(y_test)
pre1=prediction(gbm_pred_pca2,y_test)
roc.perf = performance(pre1, measure = "tpr", x.measure = "fpr")
auc.train= performance(pre1,measure = "auc")
auc.train = auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .2, y = .8,paste("AUC = ", round(auc.train[[1]],3), sep = ""))
submission1=data.frame(session_ID,gbm_pred_pca)

glm_pca=caret::train(is_click~.,data=train2,method='glm',trControl = tune_control, tuneLength=3)
glm_pred_pca=predict(glm_pca,newdata = test2)
confusionMatrix(glm_pred_pca,test2$is_click)

######neural net
modelLookup(model = "nnet")
nn_pca=caret::train(is_click~.,data=train2,
                    method='nnet',trControl = tune_control, tuneLength=3)
nn_pred_pca=predict(nn_pca,newdata = test2)
confusionMatrix(nn_pred_pca,test2$is_click)

library(ROCR)
class(test_y)
nn_pred_pca=as.numeric(nn_pred_pca)
test_y=as.numeric(test_y)
head(test_y)
nn.pre1=prediction(nn_pred_pca,test_y)
roc.perf = performance(nn.pre1, measure = "tpr", x.measure = "fpr")
auc.train = performance(nn.pre1,measure = "auc")
auc.train = auc.train@y.values
plot(roc.perf)
abline(a=0, b= 1)
text(x = .2, y = .8,paste("AUC = ", round(auc.train[[1]],3), sep = ""))

########################3
