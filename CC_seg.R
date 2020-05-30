#This case requires trainees to develop a customer segmentation to define marketing strategy.

#clearing all the existing objects
rm(list=ls())

#set working directory
setwd("C:/Users/Sunny/Desktop/edwisor/Credit Card")
getwd()


#install basic required packages
#install.packages(c(

#reading the data file
Credit_Card = read.csv("credit-card-data.csv", header = TRUE, sep = ",")

#sample data 
head(Credit_Card)

#dimensions of data
dim(Credit_Card)

#Summarty of the data
summary(Credit_Card)

#structure of the data
str(Credit_Card)

#We can see that the given data only consists of numeric data and only categorical 
#vaiable present is CUST_ID which we will anyways omit as its of no use for analysis and grouping.

#removing CUST_ID from the data
Credit_Card = Credit_Card[-1]
names(Credit_Card)

#some variables are of range 0 to 1 and few are of the range 0 to 30000, 
#hence data scaling is important for grouping the given data


#finding the missing values
missing_val = data.frame(apply(Credit_Card,2,function(x){sum(is.na(x))}))
missing_val

#We can see that the variables CREDIT_LIMIT and MINIMUM_PAYMENTS has missing values

#lets find percentage of missing values
install.packages("tidyr")
install.packages("DataExplorer")
library("tidyr")
library("DataExplorer")
plot_missing(Credit_Card)

#from the plot we can see that only one value missing from CREDIT_LIMIT and its 0.01% and then 
#MINIMUM_PAYMENTS has 3.5% o missing values. Which means its better to 
#impute the values using mean value of that variable in this case


#Treating missing values:
Credit_Card$CREDIT_LIMIT[which(is.na(Credit_Card$CREDIT_LIMIT))] <- mean(Credit_Card$CREDIT_LIMIT, na.rm=TRUE) 

Credit_Card$MINIMUM_PAYMENTS[which(is.na(Credit_Card$MINIMUM_PAYMENTS))] <- mean(Credit_Card$MINIMUM_PAYMENTS, na.rm=TRUE) 

plot_missing(Credit_Card)

#Now there are no missing values


#Data Scalaing:(standardization)

Credit_Card1 = scale(Credit_Card)
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
Credit_Card <- normalize(Credit_Card)

summary(Credit_Card1)


#applying PCA:
#install.packages("factoextra")
#library(factoextra)
#Credit_Card.PCA <- prcomp(Credit_Card, center = TRUE,scale. = TRUE)
#summary(Credit_Card.PCA)

#fviz_eig(Credit_Card.PCA)

#up to PC8 explains about 85% varience hence retaining upto PC8 
#PCA_CC_data = data.frame(Credit_Card.PCA$x[,1:8])
#summary(PCA_CC_data)


#K-means cluestering:

#finding no of clusters to build using elbow graph
set.seed(123)
# Compute and plot wss for k = 2 to k = 15.
k.max <- 15
data <- Credit_Card1
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

#using the elbow graph we can see that  no of clusters to be 4



#K-means clustering
kmeans_model = kmeans(data, 8, nstart = 50, iter.max = 15)
#we keep number of iter.max=15 toa ensure the algorithm converges and nstart=50 to 
#ensure that atleat 50 random sets are choosen  
#As the final result of k-means clustering result is sensitive to the 
#random starting assignments, we specify nstart = 25. 
#This means that R will try 25 different random starting assignments 
#and then select the best results corresponding to the one with the 
#lowest within cluster variation. 

#Summarize cluster output
kmeans_model

#CLuster analysis:

Cluster_data = cbind(Credit_Card, clusterID = kmeans_model$cluster)
Cluster_data = data.frame(Cluster_data)
head(Cluster_data)

install.packages("cluster")
library(cluster)
clusplot(Credit_Card, kmeans_model$cluster, color=TRUE, shade=TRUE,labels=2, lines=0)

#plotting obtained clusters on existing data:

install.packages("ggplot2")
library("ggplot2")

p1<-ggplot(Cluster_data, aes(x = clusterID, y = PURCHASES)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = PURCHASES), vjust = -0.3)
p2<-ggplot(Cluster_data, aes(x = clusterID, y = ONEOFF_PURCHASES)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = ONEOFF_PURCHASES), vjust = -0.3)
p3<-ggplot(Cluster_data, aes(x = clusterID, y = INSTALLMENTS_PURCHASES)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = INSTALLMENTS_PURCHASES), vjust = -0.3)
p4<-ggplot(Cluster_data, aes(x = clusterID, y = CREDIT_LIMIT)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = CREDIT_LIMIT), vjust = -0.3)
p5<-ggplot(Cluster_data, aes(x = clusterID, y = PAYMENTS)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = PAYMENTS), vjust = -0.3)
p6<-ggplot(Cluster_data, aes(x = clusterID, y = PRC_FULL_PAYMENT)) +
  geom_bar(fill = "#0073C2FF", stat = "identity") +
  geom_text(aes(label = PRC_FULL_PAYMENT), vjust = -0.3)

gridExtra::grid.arrange(p1,p2,p3,p4,p5,p6, ncol=3)


#from the barplots we can conclude the followin:

# out of 8 cluesters, cluesters like 1,3,8 areof hign purchase and they tend to buy 
# both one-off purhase and installment purchases high but they are good at 
# insallment payments but not one off payments. SO its better offer them good plans 
# for installment payments

# clusters like 1,3,6,7 are having high credit limit and we can see that people who have
# high purchases also has high paymentgs and full payments indirectly showing that high 
# credit limit comes with good payment history

# on the other hand from cluester like 2,4,5,8 we can see that people with low credit limit
# and likely to go for one-off purchase over installments and its commonsense to choose 
# one off purchases when you have low credit limit as installments need mostly need high 
# credit limit for bigger purchases