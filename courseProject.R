setwd("/Users/toddrimes/Desktop/coursera/PracticalMachineLearning/CourseProject")

runAll <- function() {
        library(caret)
        library(gbm)
        library(AppliedPredictiveModeling)
        library(rpart)
        library(doParallel)
        #Find out how many cores are available (if you don't already know)
        cores<-detectCores()
        #Create cluster with desired number of cores, leave one open for the machine         
        #core processes
        cl <- makeCluster(cores[1]-1)
        #Register cluster
        registerDoParallel(cl)
        control <- trainControl(method="repeatedcv", number=10, repeats=3)
        seed <- 7
        set.seed(seed)
        metric <- "Accuracy"
        
        training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
        projectTesting <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
        
        # CROSS VALIDATION
        splitTrain <- createDataPartition(y=training$classe, p=0.8, list=FALSE)
        myTraining <- training[splitTrain, ]
        myTesting <- training[-splitTrain, ]
        
        # Remove columns that are majority NA
        myTraining <- myTraining[,colSums(is.na(myTraining))<nrow(myTraining) * 0.6]
        myTesting <- myTesting[,colSums(is.na(myTesting))<nrow(myTesting) * 0.6]
        projectTesting <- projectTesting[,colSums(is.na(projectTesting))<nrow(projectTesting) * 0.6]
        
        # Discover which columns have near-zero-variance
        myDataNZV <- nearZeroVar(myTraining, saveMetrics=TRUE)
        myDataNZV <- subset(myDataNZV,nzv==TRUE)
        removeColNames <- rownames(myDataNZV)
        
        # Remove the near-zero-variance columns from the training, testing, and validation data sets
        myTraining <- myTraining[, -which(names(myTraining) %in% removeColNames)]
        myTraining <- myTraining[, 7:59]
        myTesting <- myTesting[, -which(names(myTesting) %in% removeColNames)]
        myTesting <- myTesting[, 7:59]
        projectTesting <- projectTesting[, 8:60]
        
        # TRAIN INDIVIDUAL MODEL FITS with 1) boosting, 2) random forests, 3) bagging, and 4) trees
        #Boosting - SAVED
        if(file.exists("fit_gbm.rda")) {
                ## load model
                load("fit_gbm.rda")
        } else {
                ## (re)fit the model
                set.seed(seed)
                fit_gbm <- train(classe~., data=myTraining, method="gbm", metric=metric, trControl=control, verbose=FALSE)
                save(fit_gbm, file="fit_gbm.rda")
        }
        if(file.exists("pred_gbm.rda")) {
                ## load model
                load("pred_gbm.rda")
        } else {
                ## (re)fit the model
                pred_gbm <- predict(fit_gbm, myTesting)
                save(pred_gbm, file="pred_gbm.rda")
        }
        # print(fit_gbm) 
        # calculate the misclassification error rate
        misclfn_gbm = mean(pred_gbm != myTraining$classe)
        #confusionMatrix(pred_gbm, myTesting$classe)$overall[1]
        
        #Random Forests - SAVED
        if(file.exists("fit_rf.rda")) {
                ## load model
                load("fit_rf.rda")
        } else {
                ## (re)fit the model
                #fit_rf <- train(classe~., data=myTraining, method="rf", metric=metric, trControl=control, importance=TRUE, verbose=FALSE)
                set.seed(seed)
                fit_rf <- train(classe~., data=myTraining, method="rf", metric=metric, trControl = trainControl(method = "oob"), importance=TRUE, verbose=FALSE)
                save(fit_rf, file="fit_rf.rda")
        }
        if(file.exists("pred_rf.rda")) {
                ## load model
                load("pred_rf.rda")
        } else {
                ## (re)fit the model
                pred_rf <- predict(fit_rf, myTesting)
                save(pred_rf, file="pred_rf.rda")
        }
        # print(fit_rf)
        # calculate the misclassification error rate
        misclfn_rf = mean(pred_rf != myTraining$classe)
        # confusionMatrix(pred_rf, myTesting$classe)$overall[1]
        
        #Bagging - SAVED
        if(file.exists("fit_bag.rda")) {
                ## load model
                load("fit_bag.rda")
        } else {
                ## (re)fit the model
                set.seed(seed)
                # cannot run parallel :~(
                cl <- makeCluster(1)
                #Register cluster of 1
                registerDoParallel(cl)
                fit_bag <- train(classe~., data=myTraining, method="treebag", metric=metric, trControl=control)
                save(fit_bag, file="fit_bag.rda")
        }
        if(file.exists("pred_bag.rda")) {
                ## load model
                load("pred_bag.rda")
        } else {
                ## (re)fit the model
                pred_bag <- predict(fit_bag, myTesting)
                save(pred_bag, file="pred_bag.rda")
        }
        # print(fit_bag)
        # calculate the misclassification error rate
        misclfn_bag = mean(pred_bag != myTraining$classe)
        # confusionMatrix(pred_bag, myTesting$classe)$overall[1]
        
        #Tree - SAVED
        if(file.exists("fit_rp.rda")) {
                ## load model
                load("fit_rp.rda")
        } else {
                ## (re)fit the model
                set.seed(seed)
                # cannot run parallel :~(
                cl <- makeCluster(1)
                #Register cluster of 1
                registerDoParallel(cl)
                fit_rp <- train(classe~., data=myTraining, method="rpart", metric=metric, trControl=control)
                save(fit_rp, file="fit_rp.rda")
        }
        if(file.exists("pred_rp.rda")) {
                ## load model
                load("pred_rp.rda")
        } else {
                ## (re)fit the model
                pred_rp <- predict(fit_rp, myTesting)
                save(pred_rp, file="pred_rp.rda")
        }
        # print(fit_rp)
        # calculate the misclassification error rate
        misclfn_rp = mean(pred_rp != myTraining$classe)
        # confusionMatrix(pred_rp, myTesting$classe)$overall[1]
        
        results <- resamples(list(rpart=fit_rp, gbm=fit_gbm, bagging=fit_bag, rf=fit_rf))
        me_list = data.frame(as.list(c(misclfn_gbm, misclfn_rf, misclfn_bag, misclfn_rp)))
        colnames(me_list) = c("Boosting","Random Forest","Bagging","Trees")
        print("Misclassification Error - by model type")
        print(me_list)
        # Table comparison
        summary(results)
        
        # boxplot comparison
        bwplot(results)
        # Dot-plot comparison
        # dotplot(results)
        
        # TEST the chosen model (Random Forests) against the original testing data for validation
        print(fit_rf)
        pred_rf2 <- predict(fit_rf, projectTesting)
        print(pred_rf2)
        fit_rf$final
        
        write_answers = function(x){
                n = length(x)
                for(i in 1:n){
                        filename = paste0("problem-",i,".txt")
                        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
                }
        }
        
        write_answers(pred_rf2)
        
        # So the expected prediction accuracy based on the witheld portion of our original training set is 0.967
        
        
}