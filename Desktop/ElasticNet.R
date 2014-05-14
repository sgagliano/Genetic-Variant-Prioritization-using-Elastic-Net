#ElasticNet.R- Version 1.0; created May 14, 2014
#Refer to Gagliano SA, Barnes MR, Weale ME, Knight J (2014) A Bayesian method to incorporate hundreds of functional characteristics with association evidence to improve variant prioritization. PLoS ONE

#This function takes two inputs, the training and test sets, called MLno1 and MLall1, respectively. 
#Both the MLno1 and MLall1 consist of 19 columns- the first 4 are the genetiv variant identifiers (rsID, BP, chr, BP), columns 5-18 are the functional characteristics and column 19 is the binary classifier (0/1)

##Notes about running this script
#make sure the glmnet package is installed before this code is run
#run this script from the directory containing the training and test sets
#source('ElasticNet.R')
#elasticnet(MLno1, MLall1)

elasticnet<-function(MLno1, MLall1){
require("glmnet")
#Prep data
MLno1<-read.table("MLno1")	        #training set	
Ghitno1 <-as.factor(MLno1$V19)		#Hits for glm & pred
MLno1M <- as.matrix(MLno1[,5:18])		#make it a matrix

MLall1<-read.table("MLall1")	        #test set
Ghitall1 <-as.numeric (MLall1$V19)		# Hits for glm & pred
MLall1M <- as.matrix(MLall1[,5:18])		#make it a matrix

#weights
c<-(length(Ghitno1))/2
WA<-c/sum(MLno1$V19) #for hits
WB<-c/(length(Ghitno1)-sum(MLno1$V19)) #for nonhits
W<-WA-WB
Ghitno1W <-MLno1$V19*W+WB #hits are assigned a weight of WA and nonhits are assigned WB

best.lambda <- rep(NA,21)
optimal.nr <- rep(NA,21)
optimal.dev <- rep(NA,21)
options(warn=2)

set.seed(100)
alpha <- seq(0.0,1,0.05)
for (i in  1:21){
	print(paste(i,":","alpha=",alpha[i]))
	try(cv.model <- cv.glmnet(MLno1M, Ghitno1,family="binomial",alpha=alpha[i],nfolds=10, weights=Ghitno1W))
	cv.lambda <- cv.model$lambda
	dev.array<- as.array(cv.model$cvm)
	ncoeff.array <- as.array(cv.model$nzero)
	best.lambda[i] <- cv.model$lambda.1se
	optimal.nr[i] <-ncoeff.array[cv.lambda==best.lambda[i]]
	optimal.dev[i] <- dev.array[cv.lambda==best.lambda[i]]
	}
options(warn=0)    #Reset warning-handling to default

write.csv(best.lambda,"clump1abest.lambda.csv")
write.csv(optimal.dev,"clump1abest.dev.csv")
write.csv(optimal.nr,"clump1abest.nr.csv")

dev<-read.table("clump1abest.dev.csv", sep=",", as.is=T, h=T)
alpha<-seq(0,1,0.05)
alpha<-as.data.frame(alpha)
alpha$dev<- dev[,2]
lambda<-read.table("clump1abest.lambda.csv", sep=",", as.is=T, h=T)
alpha$lambda<- lambda[,2]
sortbydev<-alpha[with(alpha, order(dev)), ]
bestalpha<-sortbydev[1,1]
bestlambda<-sortbydev[1,3]
print("alpha is")
print(bestalpha)
print("lambda is")
print(bestlambda)

fixedLASSO1a<- glmnet(MLno1M, Ghitno1, family="binomial", alpha=paste(bestalpha), lambda=paste(bestlambda), weights=Ghitno1W)
write.csv(fixedLASSO1a$beta[,1],"fixedLASSO1a.csv")
print("Beta coefficients written to fixedLASSO1a.csv")


Predictions1aTr <- predict(fixedLASSO1a, MLno1M,type="response",exact=FALSE)
Predictions1aTe <- predict(fixedLASSO1a, MLall1M,type="response",exact=FALSE)


Test<-cbind(MLall1, Predictions1aTe)
Train<-cbind(MLno1, Predictions1aTr)
write.table(Train, "trainset.csv", sep=",", col.names=F, row.names=F, quote=F)
write.table(Test, "testset.csv", sep=",", col.names=F, row.names=F, quote=F)
print("Training set results written to trainset.csv")
print("Test set results written to testset.csv")
}
