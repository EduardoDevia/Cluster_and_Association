
#install.packages("FSelector")
library(grid)
library(partykit)
library(RWeka)
library(partykit)
library(FSelector)
library(e1071)
library(lattice)
library(ggplot2)
library(caret)
library(RWekajars)
#help(make_Weka_associator)
#WOW("M5P")
#Load the arff file
setwd("C:/Users/chedevia.NA/Downloads")
insuarance_train <- read.csv("train.csv", stringsAsFactors = FALSE)
insuarance_train[insuarance_train=="-1"]<-NA
insuarance_test <- read.csv("test.csv", stringsAsFactors = FALSE)
insuarance_test[insuarance_test=="-1"]<-NA
insuarance_train_clean<-na.omit(insuarance_train)
insuarance_test_clean<-na.omit(insuarance_test)
insuarance_train_clean$target<-as.factor(insuarance_train_clean$target)
summary(insuarance_train_clean$target)/(119261+5670)
summary(insuarance_train$target)/(573518+21694)
write.table(insuarance_train_clean, file = "~/insuarance_train_clean.csv", sep = ",", col.names = NA, qmethod = "double")                  
#===============Building the Models===================
#Numeric
functions
MultilayerPerceptron
SMOreg
#IBk
Bagging
DecisionTable
M5Rules
ZeroR
DecisionStump
M5P
RamdonForest
REPTree
#================Classification==============#
#BayesNet
#naiveBayes
#Logistic
#MultilayerPerceptron
#SMO
#Bagging
#LogitBoost
DecistionTable
#OneR
Part
#ZeroR
#DesicionStump
#J48
#LMT
#randomForest         ----------Java.lang.OutOfMemoryError
#Randomtree
#REPTree

#===============ZeroR===================
ZeroR<-make_Weka_classifier("weka/classifiers/rules/ZeroR")
ZeroR_Classifier<-ZeroR(insuarance_train_clean$target~ ., data = insuarance_train_clean)
ZeroR_Train<-summary(ZeroR_Classifier)
ZeroR_true<-ZeroR_Train$confusionMatrix[2,2]
ZeroR_true_Accuracy<-(ZeroR_Train$confusionMatrix[2,2]+ZeroR_Train$confusionMatrix[1,1])/124931
#Cross Validation
ZeroR_CV <- evaluate_Weka_classifier(ZeroR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
ZeroR_true_Accuracy_CV<-(ZeroR_CV$confusionMatrix[2,2]+ZeroR_CV$confusionMatrix[1,1])/124931
ZeroR_true_CV<-ZeroR_CV$confusionMatrix[2,2]
#===============MultiLayerPerceptron===================
MultilayerPerceptron<-make_Weka_classifier("weka/classifiers/functions/MultilayerPerceptron")
MultilayerPerceptron_Classifier<-MultilayerPerceptron(insuarance_train_clean$target~ ., data = insuarance_train_clean)
MultilayerPerceptron_Train<-summary(MultilayerPerceptron_Classifier)
MultilayerPerceptron_true<-MultilayerPerceptron_Train$confusionMatrix[2,2]
MultilayerPerceptron_true_Accuracy<-(MultilayerPerceptron_Train$confusionMatrix[2,2]+MultilayerPerceptron_Train$confusionMatrix[1,1])/124931
#Cross Validation
MultilayerPerceptron_CV <- evaluate_Weka_classifier(MultilayerPerceptron_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
MultilayerPerceptron_true_Accuracy_CV<-(MultilayerPerceptron_CV$confusionMatrix[2,2]+MultilayerPerceptron_CV$confusionMatrix[1,1])/124931
MultilayerPerceptron_true_CV<-MultilayerPerceptron_CV$confusionMatrix[2,2]
#===============OneR===================

OneR_Classifier<-OneR(insuarance_train_clean$target~ ., data = insuarance_train_clean)
OneR_Train<-summary(OneR_Classifier)
OneR_true<-OneR_Train$confusionMatrix[2,2]
#Cross Validation
OneR_CV <- evaluate_Weka_classifier(OneR_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
OneR_true_CV<-OneR_CV$confusionMatrix[2,2]

#===============J48===================
J48_Classifier<-J48(insuarance_train_clean$target~ ., data = insuarance_train_clean)
J48_Train<-summary(J48_Classifier)
J48_true<-J48_Train$confusionMatrix[2,2]
#Cross Validation
J48_CV <- evaluate_Weka_classifier(J48_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
J48_true_CV<-J48_CV$confusionMatrix[2,2]
#===============IBk===================
IBk_Classifier<-IBk(insuarance_train_clean$target~ ., data = insuarance_train_clean,control=Weka_control(K=1))
IBK_Train<-summary(IBk_Classifier)
IBK_true_Accuracy<-(IBK_Train$confusionMatrix[2,2]+IBK_Train$confusionMatrix[1,1])/124931
IBK_true<-IBK_Train$confusionMatrix[2,2]
#Cross Validation
IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
IBK_true_CV<-IBk_CV$confusionMatrix[2,2]
IBK_true_Accuracy_CV<-(IBk_CV$confusionMatrix[2,2]+IBk_CV$confusionMatrix[1,1])/124931

#===============BayesNet===================
BayesNet<-make_Weka_classifier("weka/classifiers/bayes/BayesNet")
BayesNet_Classifier<-BayesNet(insuarance_train_clean$target~ ., data = insuarance_train_clean)
BayesNet_Train<-summary(BayesNet_Classifier)
BayesNet_true_Accuracy<-(BayesNet_Train$confusionMatrix[2,2]+BayesNet_Train$confusionMatrix[1,1])/124931
BayesNet_true<-BayesNet_Train$confusionMatrix[2,2]
#Cross Validation
BayesNet_CV <- evaluate_Weka_classifier(BayesNet_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
BayesNet_true_CV<-BayesNet_CV$confusionMatrix[2,2]
BayesNet_true_Accuracy_CV<-(BayesNet_CV$confusionMatrix[2,2]+IBk_CV$confusionMatrix[1,1])/124931


#===============NaiveBayes===================
NaiveBayes<-make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
NaiveBayes_Classifier<-NaiveBayes(insuarance_train_clean$target~ ., data = insuarance_train_clean)
NaiveBayes_Train<-summary(NaiveBayes_Classifier)
NaiveBayes_true_Accuracy<-(NaiveBayes_Train$confusionMatrix[2,2]+NaiveBayes_Train$confusionMatrix[1,1])/124931
NaiveBayes_true<-NaiveBayes_Train$confusionMatrix[2,2]
#Cross Validation
NaiveBayes_CV <- evaluate_Weka_classifier(NaiveBayes_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
NaiveBayes_true_CV<-NaiveBayes_CV$confusionMatrix[2,2]
NaiveBayes_true_Accuracy_CV<-(NaiveBayes_CV$confusionMatrix[2,2]+IBk_CV$confusionMatrix[1,1])/124931


#===============Logistic===================
Logistic_Classifier<-Logistic(insuarance_train_clean$target~ ., data = insuarance_train_clean)
Logistic_Train<-summary(Logistic_Classifier)
Logistic_true_Accuracy<-(Logistic_Train$confusionMatrix[2,2]+Logistic_Train$confusionMatrix[1,1])/124931
Logistic_true<-Logistic_Train$confusionMatrix[2,2]
#Cross Validation
Logistic_CV <- evaluate_Weka_classifier(Logistic_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
Logistic_true_CV<-Logistic_CV$confusionMatrix[2,2]
Logistic_true_Accuracy_CV<-(Logistic_CV$confusionMatrix[2,2]+IBk_CV$confusionMatrix[1,1])/124931

#===============SMO===================
SMO_Classifier<-SMO(insuarance_train_clean$target~ ., data = insuarance_train_clean)
SMO_Train<-summary(SMO_Classifier)
SMO_true_Accuracy<-(SMO_Train$confusionMatrix[2,2]+SMO_Train$confusionMatrix[1,1])/124931
SMO_true<-SMO_Train$confusionMatrix[2,2]
#Cross Validation
SMO_CV <- evaluate_Weka_classifier(SMO_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
SMO_true_CV<-SMO_CV$confusionMatrix[2,2]
SMO_true_Accuracy_CV<-(SMO_CV$confusionMatrix[2,2]+IBk_CV$confusionMatrix[1,1])/124931

#===============LMT===================
LMT_Classifier<-LMT(insuarance_train_clean$target~ ., data = insuarance_train_clean)
LMT_Train<-summary(LMT_Classifier)
LMT_true_Accuracy<-(LMT_Train$confusionMatrix[2,2]+LMT_Train$confusionMatrix[1,1])/124931
LMT_true<-LMT_Train$confusionMatrix[2,2]
#Cross Validation
LMT_CV <- evaluate_Weka_classifier(LMT_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
LMT_true_CV<-LMT_CV$confusionMatrix[2,2]
LMT_true_Accuracy_CV<-(LMT_CV$confusionMatrix[2,2]+LMT_CV$confusionMatrix[1,1])/124931

#===============SMO===================
SMO_Classifier<-SMO(insuarance_train_clean$target~ ., data = insuarance_train_clean)
SMO_Train<-summary(SMO_Classifier)
SMO_true_Accuracy<-(SMO_Train$confusionMatrix[2,2]+SMO_Train$confusionMatrix[1,1])/124931
SMO_true<-SMO_Train$confusionMatrix[2,2]
#Cross Validation
SMO_CV <- evaluate_Weka_classifier(SMO_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
SMO_true_CV<-SMO_CV$confusionMatrix[2,2]
SMO_true_Accuracy_CV<-(SMO_CV$confusionMatrix[2,2]+SMO_CV$confusionMatrix[1,1])/124931


#===============RandomForest===================
RandomForest<-make_Weka_classifier("weka/classifiers/trees/RandomForest")
RandomForest_Classifier<-RandomForest(insuarance_train_clean$target~ ., data = insuarance_train_clean)
RandomForest_Train<-summary(RandomForest_Classifier)
RandomForest_true_Accuracy<-(RandomForest_Train$confusionMatrix[2,2]+RandomForest_Train$confusionMatrix[1,1])/124931
RandomForest_true<-RandomForest_Train$confusionMatrix[2,2]
#Cross Validation
RandomForest_CV <- evaluate_Weka_classifier(RandomForest_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
RandomForest_true_CV<-RandomForest_CV$confusionMatrix[2,2]
RandomForest_true_Accuracy_CV<-(RandomForest_CV$confusionMatrix[2,2]+IBk_CV$confusionMatrix[1,1])/124931

#===============RandomTree===================
RandomTree<-make_Weka_classifier("weka/classifiers/trees/RandomTree")
RandomTree_Classifier<-RandomTree(insuarance_train_clean$target~ ., data = insuarance_train_clean)
RandomTree_Train<-summary(RandomTree_Classifier)
RandomTree_true_Accuracy<-(RandomTree_Train$confusionMatrix[2,2]+RandomTree_Train$confusionMatrix[1,1])/124931
RandomTree_true<-RandomTree_Train$confusionMatrix[2,2]
#Cross Validation
RandomTree_CV <- evaluate_Weka_classifier(RandomTree_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
RandomTree_true_CV<-RandomTree_CV$confusionMatrix[2,2]
RandomTree_true_Accuracy_CV<-(RandomTree_CV$confusionMatrix[2,2]+RandomTree_CV$confusionMatrix[1,1])/124931



#===============REPTree===================
REPTree<-make_Weka_classifier("weka/classifiers/trees/REPTree")
REPTree_Classifier<-REPTree(insuarance_train_clean$target~ ., data = insuarance_train_clean)
REPTree_Train<-summary(REPTree_Classifier)
REPTree_true_Accuracy<-(REPTree_Train$confusionMatrix[2,2]+REPTree_Train$confusionMatrix[1,1])/124931
REPTree_true<-REPTree_Train$confusionMatrix[2,2]
#Cross Validation
REPTree_CV <- evaluate_Weka_classifier(REPTree_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
REPTree_true_CV<-REPTree_CV$confusionMatrix[2,2]
REPTree_true_Accuracy_CV<-(REPTree_CV$confusionMatrix[2,2]+REPTree_CV$confusionMatrix[1,1])/124931
RWeka::d
#===============DecisionStump===================
DecisionStump_Classifier<-DecisionStump(insuarance_train_clean$target~ ., data = insuarance_train_clean)
DecisionStump_Train<-summary(DecisionStump_Classifier)
DecisionStump_true_Accuracy<-(DecisionStump_Train$confusionMatrix[2,2]+DecisionStump_Train$confusionMatrix[1,1])/124931
DecisionStump_true<-DecisionStump_Train$confusionMatrix[2,2]
#Cross Validation
DecisionStump_CV <- evaluate_Weka_classifier(DecisionStump_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
DecisionStump_true_CV<-DecisionStump_CV$confusionMatrix[2,2]
DecisionStump_true_Accuracy_CV<-(DecisionStump_CV$confusionMatrix[2,2]+DecisionStump_CV$confusionMatrix[1,1])/124931


#===============LogitBoost===================
LogitBoost_Classifier<-LogitBoost(insuarance_train_clean$target~ ., data = insuarance_train_clean)
LogitBoost_Train<-summary(LogitBoost_Classifier)
LogitBoost_true_Accuracy<-(LogitBoost_Train$confusionMatrix[2,2]+LogitBoost_Train$confusionMatrix[1,1])/124931
LogitBoost_true<-LogitBoost_Train$confusionMatrix[2,2]
#Cross Validation
LogitBoost_CV <- evaluate_Weka_classifier(LogitBoost_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
LogitBoost_true_CV<-LogitBoost_CV$confusionMatrix[2,2]
LogitBoost_true_Accuracy_CV<-(LogitBoost_CV$confusionMatrix[2,2]+LogitBoost_CV$confusionMatrix[1,1])/124931




#===============PART===================
PART_Classifier<-PART(insuarance_train_clean$target~ ., data = insuarance_train_clean)
PART_Train<-summary(PART_Classifier)
PART_true_Accuracy<-(PART_Train$confusionMatrix[2,2]+PART_Train$confusionMatrix[1,1])/124931
PART_true<-PART_Train$confusionMatrix[2,2]
#Cross Validation
PART_CV <- evaluate_Weka_classifier(PART_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
PART_true_CV<-PART_CV$confusionMatrix[2,2]
PART_true_Accuracy_CV<-(PART_CV$confusionMatrix[2,2]+PART_CV$confusionMatrix[1,1])/124931

#===============Bagging===================
Bagging_Classifier<-Bagging(insuarance_train_clean$target~ ., data = insuarance_train_clean)
Bagging_Train<-summary(Bagging_Classifier)
Bagging_true_Accuracy<-(Bagging_Train$confusionMatrix[2,2]+Bagging_Train$confusionMatrix[1,1])/124931
Bagging_true<-Bagging_Train$confusionMatrix[2,2]
#Cross Validation
Bagging_CV <- evaluate_Weka_classifier(Bagging_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
Bagging_true_CV<-Bagging_CV$confusionMatrix[2,2]
Bagging_true_Accuracy_CV<-(Bagging_CV$confusionMatrix[2,2]+Bagging_CV$confusionMatrix[1,1])/124931
WOW(Bagging)

#
memory.limit(size=TRUE)
memory.size(TRUE)
memory.size()
#




















#===============IBk C & M Optimisation===================
WOW(IBk)
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Knew<-0
Neighbour<-0
df_IBk <- data.frame(SAMPLE=double(),O_F=double(),OF_New=double(),K_New=double(),CV_New=double(),"OFnew+CVnew"=double(),T_R=double())
for (i in 1:20) {
  Neighbour=i
  IBk_Classifier<-IBk(insuarance_train_clean$target~ ., data = insuarance_train_clean,control=Weka_control(K=Neighbour))
  IBk_Classifier_Summary<-summary(IBk_Classifier)
  TR<-IBk_Classifier_Summary$details[[1]]
  
  #Cross Validation
  IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-IBk_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Knew,Knew<-Neighbour)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  df_IBk[nrow(df_IBk) + 1,] = list(SAMPLE=i,O_F=OF,OF_New=OFnew,K_New=Knew,CV_New=CVnew,"OFnew+CVnew"=(CVnew-(OFnew*CVnew)),T_R=TR)
  
}
#===============J48 C & M Optimisation===================
WOW(J48)

OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
for (i in 1:100) {
  confidence<-runif(1, 0, 0.25)
  minObjleaf<-floor(runif(1, 1,10)) 
  J48_Classifier<-J48(insuarance_train_clean$target~ ., data = insuarance_train_clean, control=Weka_control(C=confidence,M=minObjleaf))
  J48_Classifier_Summary<-summary(J48_Classifier)
  TR<-J48_Classifier_Summary$details[[1]]
  
  #Cross Validation
  J48_CV <- evaluate_Weka_classifier(J48_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  CV<-J48_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Cnew,Cnew<-confidence)
  ifelse(-OFnew+CVnew>-OF+CV,Mnew,Mnew<-minObjleaf)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  print(paste0(" OF ", format(round(OF,digits = 4), nsmall = 4)," OF New :", format(round(OFnew,digits = 4), nsmall = 4), " CNew :", 
               format(round(Cnew,digits = 4), nsmall = 4), " MNew :", format(round(Mnew,digits = 4), nsmall = 4),"  CV New ", 
               format(round(CV,digits = 4), nsmall = 4),"  OFnew+CVnew :", format(round(OFnew+CVnew,digits = 4), nsmall = 4),"  TR New :", format(round(TR,digits = 4), nsmall = 4)
  ) )
}
#=================Optimizing the attributes=================
GainRatioAttributeEval(insuarance_train_clean$target~ . , data = insuarance_train_clean)
insuarance_train_clean_GainR<-subset(insuarance_train_clean, select = c(-1,-2))
InfoGainAttributeEval(insuarance_train_clean$target~ . , data = insuarance_train_clean)
insuarance_train_clean_InfoGain<-subset(insuarance_train_clean, select = c(-1,-6,-17))

#==================Evaluate with new set InfoGain====================
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
for (i in 1:20) {
  Neighbour<-i+1
  
  IBk_Classifier<-IBk(insuarance_train_clean_InfoGain$class~ ., data = insuarance_train_clean_InfoGain,control=Weka_control(K=Neighbour))
  IBk_Classifier_Summary<-summary(IBk_Classifier)
  TR<-IBk_Classifier_Summary$details[[1]]
  
  #Cross Validation
  IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-IBk_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Knew,Knew<-Neighbour)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  print(paste0(" OF ", format(round(OF,digits = 4), nsmall = 4)," OF New :", format(round(OFnew,digits = 4), nsmall = 4), " KNew :", 
               format(round(Knew,digits = 4), nsmall = 4), "  CV New ", format(round(CV,digits = 4), nsmall = 4),"  OFnew+CVnew :", format(round(OFnew+CVnew,digits = 4), nsmall = 4),"  TR New :", format(round(TR,digits = 4), nsmall = 4)
               ,"Value ",i) )
}

#==================Evaluate with new set InfoGain====================
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
for (i in 1:20) {
  Neighbour<-i+1
  
  IBk_Classifier<-IBk(insuarance_train_clean_GainR$class~ ., data = insuarance_train_clean_GainR,control=Weka_control(K=Neighbour))
  IBk_Classifier_Summary<-summary(IBk_Classifier)
  TR<-IBk_Classifier_Summary$details[[1]]
  
  #Cross Validation
  IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-IBk_CV$details[[1]]
  OF<-TR-CV
  ifelse(-OFnew+CVnew>-OF+CV,OFnew,OFnew<-OF)
  ifelse(-OFnew+CVnew>-OF+CV,Knew,Knew<-Neighbour)
  ifelse(-OFnew+CVnew>-OF+CV,CVnew,CVnew<-CV)
  print(paste0(" OF ", format(round(OF,digits = 4), nsmall = 4)," OF New :", format(round(OFnew,digits = 4), nsmall = 4), " KNew :", 
               format(round(Knew,digits = 4), nsmall = 4), "  CV New ", format(round(CV,digits = 4), nsmall = 4),"  OFnew+CVnew :", format(round(OFnew+CVnew,digits = 4), nsmall = 4),"  TR New :", format(round(TR,digits = 4), nsmall = 4)
               ,"Value ",i) )
}

#======================Part 2 Numberic Prediction=====================
#===============  M5P===================
M5P_Classifier<-M5P(insuarance_train_clean$age~ ., data = insuarance_train_clean)
summary(M5P_Classifier)
#Cross Validation
M5P_CV <- evaluate_Weka_classifier(M5P_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
M5P_CV

#===============IBk===================
IBk_Classifier<-IBk(insuarance_train_clean$age~ ., data = insuarance_train_clean,control=Weka_control(K=1))
summary(IBk_Classifier)
#Cross Validation
IBk_CV <- evaluate_Weka_classifier(IBk_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
IBk_CV
WOW("M5P")

#===============  M5P improving ===================

df <- data.frame(SAMPLE=double(),O_F=double(),OF_New=double(),M_New=double(),CV_New=double(),"OFnew+CVnew"=double(),T_R=double())
OF<-0
CVnew<-0
OFnew<-0
Cnew<-0
Mnew<-0
TR<-0
MinInstances<-0
for (i in 1:50) {
  MinInstances<-i+1
  M5P_Classifier<-M5P(insuarance_train_clean$age~ ., data = insuarance_train_clean, control=Weka_control(M=MinInstances,U=FALSE))
  M5P_Classifier_Summary<-summary(M5P_Classifier)
  TR<-M5P_Classifier_Summary$details[[1]]
  
  #Cross Validation
  M5P_CV <- evaluate_Weka_classifier(M5P_Classifier, numFolds = 10, complexity = FALSE, seed = 1, class = TRUE)
  #Cross Validation
  CV<-M5P_CV$details[[1]]
  OF<-TR-CV
  ifelse(CVnew-(OFnew*CVnew)>CV-(OF*CV),OFnew,OFnew<-OF)
  ifelse(CVnew-(OFnew*CVnew)>CV-(OF*CV),Mnew,Mnew<-MinInstances)
  ifelse(CVnew-(OFnew*CVnew)>CV-(OF*CV),CVnew,CVnew<-CV)
  df[nrow(df) + 1,] = list(SAMPLE=i,O_F=OF,OF_New=OFnew,M_New=Mnew,CV_New=CVnew,"OFnew+CVnew"=(CVnew-(OFnew*CVnew)),T_R=TR)
   
}

#==============Plotting the data====================
library(ggplot2)
library(reshape2)
df1 <-subset(df, select = c(-4))
df1 <- melt(df1, id.vars="SAMPLE")
# Everything on the same plot with smooth stat
ggplot(df1, aes(SAMPLE,value, col=variable)) + 
  geom_point() + 
  stat_smooth() 
# Everything on the same plot normal data
ggplot(df1, aes(SAMPLE,value, color = variable)) +
  theme_bw() +
  geom_line()
# Individuals Graphs
ggplot(df1, aes(SAMPLE,value)) +
  theme_bw() +
  geom_line() +
  facet_wrap(~ variable)


#======================Part 3 Clustering=====================
insuarance_train_clean_Clustering<-subset(insuarance_train_clean_Original,select=c(1,2,3,4,11))
#===============  kmeans ===================
SKmeans <- SimpleKMeans(insuarance_train_clean_Clustering, Weka_control(N = 10))
#SKmeans <- SimpleKMeans(insuarance_train_clean_Clustering[, -1], Weka_control(N = 1))
SKmeans
#table(predict(SKmeans), insuarance_train_clean_Clustering$age)
WOW(SimpleKMeans)
#===============  kmeans <> # seeds ===================
SKmeans1 <- SimpleKMeans(insuarance_train_clean_Clustering, Weka_control(N = 10,S=1))
SKmeans10 <- SimpleKMeans(insuarance_train_clean_Clustering, Weka_control(N = 10,S=10))
SKmeans100 <- SimpleKMeans(insuarance_train_clean_Clustering, Weka_control(N = 10,S=100))
SKmeans1000 <- SimpleKMeans(insuarance_train_clean_Clustering, Weka_control(N = 10,S=1000))
SKmeans1
SKmeans10
SKmeans100
SKmeans1000


#============== Association ======================
bakerydata1<-read.arff("C:/Users/chedevia.NA/Documents/bakery-data1.arff")
WOW(Apriori)
#============== Apriori ==========================
#============== Confidence =======================
Apriori(bakerydata1, Weka_control(N=10,T=0,C=0.9,D=0.05,U=0.9))
#============== Lift =======================
Apriori(bakerydata1, Weka_control(N=10,T=1,C=1.1,D=0.05,U=0.9))
#============== Lift =======================
Apriori(bakerydata1, Weka_control(N=10,T=3,C=1.1,D=0.05,U=0.9))
#============== FPGrowth ===================
install.packages("RKEEL")
library(RKEEL)
Bakery<-bakerydata1
Bakery<-loadKeelDataset("iris")
algorithm <-FPgrowth_A(Bakery,MinimumSupport = 0.9,MinimumConfidence = 0.9)
#Run algorithm
algorithm$run()
#Rules in format arules
algorithm$rules
#Show a number of rules
algorithm$showRules(2)
#Return a data.frame with all interest measures of set rules
algorithm$getInterestMeasures()
#Add interst measure YuleY to set rules
algorithm$addInterestMeasure("YuleY","yulesY")
#Sort by interest measure lift
algorithm$sortBy("lift")
#Save rules in CSV file
algorithm$writeCSV("myrules")
#Save rules in PMML file
algorithm$writePMML("myrules")

