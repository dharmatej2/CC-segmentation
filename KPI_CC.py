
# coding: utf-8

# In[451]:

#lets import required packages:
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')
import seaborn as sns


# In[452]:

#lets Set working directory
os.chdir("C:\\Users\\Sunny\\Desktop\\edwisor\\Credit Card")
os.getcwd()


# In[453]:

# reading data into dataframe
Credit_card= pd.read_csv("credit-card-data.csv")


# In[454]:

#sample data
Credit_card.head(5)


# In[455]:

#data types of variables
Credit_card.dtypes


# In[456]:

#no.of rows and columns
Credit_card.shape


# In[457]:

#Summary of the data
Credit_card.describe()


# In[458]:

#lets find the missing values in the data
print(Credit_card.isnull().sum())


# In[459]:

# we can see that the variables "CREDIT_LIMIT" and "MINIMUM_PAYMENTS" has missing values


# In[460]:

#treating missing values
#impute the missing values using the respectable variable mean
Credit_card['CREDIT_LIMIT'].fillna(Credit_card['CREDIT_LIMIT'].mean(),inplace=True)
Credit_card['MINIMUM_PAYMENTS'].fillna(Credit_card['MINIMUM_PAYMENTS'].mean(),inplace=True)
print (Credit_card.isnull().sum())


# In[461]:

#after treating missing values we can see that there are no mising values in the data


# In[462]:

#lets drop the forst column "CUST_ID" as it will not help in analysis
Credit_card1 = Credit_card.drop(['CUST_ID'], axis = 1) 


# In[463]:

Credit_card1.head(3)


# In[464]:

#lets plot some variables and try getting some useful insights


# In[465]:

#Percentage oneoff purchases:
Credit_card1['Percentage_oneoff_purchases']=(Credit_card1['ONEOFF_PURCHASES']/Credit_card1['PURCHASES'])*100
Credit_card1['Percentage_oneoff_purchases'].head(5)


# In[466]:

Credit_card1.plot.scatter(x='PURCHASES', y='Percentage_oneoff_purchases')


# In[467]:

#Percentage_Installment_purchase:
Credit_card1['Percentage_Installment_purchase']=(Credit_card1['INSTALLMENTS_PURCHASES']/Credit_card1['PURCHASES'])*100
Credit_card1['Percentage_Installment_purchase'].head(5)


# In[468]:

Credit_card1.plot.scatter(x='PURCHASES', y='Percentage_Installment_purchase')


# In[469]:

#Percentage_Usage
Credit_card1['Percentage_Usage']=(Credit_card1['PURCHASES']/Credit_card1['CREDIT_LIMIT'])*100
Credit_card1['Percentage_Usage'].head(5)


# In[470]:

Credit_card1.plot.scatter(x='Percentage_Usage', y='CREDIT_LIMIT')


# In[471]:

#Cash_Advance_percentage:
Credit_card1['Cash_Adv_prcentage']=(Credit_card1['CASH_ADVANCE']/Credit_card1['CREDIT_LIMIT'])*100
Credit_card1['Cash_Adv_prcentage'].head(5)


# In[472]:

Credit_card1.plot.scatter(x='CREDIT_LIMIT', y='Cash_Adv_prcentage')


# In[473]:

#Monthly_average_purchase:
Credit_card1['Monthly_avg_purchase']=Credit_card1['PURCHASES']/Credit_card1['TENURE']
Credit_card1['Monthly_avg_purchase'].head(5)


# In[474]:

Credit_card1.plot.scatter(x='TENURE', y='Monthly_avg_purchase')


# In[475]:

#Monthly_cash_advance:
Credit_card1['Monthly_cash_advance']=Credit_card1['CASH_ADVANCE']/Credit_card1['TENURE']
Credit_card1['Monthly_cash_advance'].head(5)


# In[476]:

Credit_card1.plot.scatter(x='TENURE', y='Monthly_cash_advance')


# In[477]:

#payments to minpayments:
Credit_card1['payment_minimumpayments']=Credit_card1.apply(lambda x:x['PAYMENTS']/x['MINIMUM_PAYMENTS'],axis=1)
Credit_card1['payment_minimumpayments'].head(5)


# In[478]:

#We can see that there are 4 types of purchase behaviors in the customers from the given data set.

#           1.People who only do One-Off Purchases.
 #          2.People who only do Installments Purchases.
  #         3.People who do both.
   #        4.People who do none.

#So deriving a categorical variable based on the customer behaviour.


# In[479]:

Credit_card1[(Credit_card1['ONEOFF_PURCHASES']==0) & (Credit_card1['INSTALLMENTS_PURCHASES']==0)].shape



# In[480]:

Credit_card1[(Credit_card1['ONEOFF_PURCHASES']>0) & (Credit_card1['INSTALLMENTS_PURCHASES']>0)].shape



# In[481]:

Credit_card1[(Credit_card1['ONEOFF_PURCHASES']>0) & (Credit_card1['INSTALLMENTS_PURCHASES']==0)].shape



# In[482]:

Credit_card1[(Credit_card1['ONEOFF_PURCHASES']==0) & (Credit_card1['INSTALLMENTS_PURCHASES']>0)].shape


# In[483]:

def purchase(Credit_card1):   
    if (Credit_card1['ONEOFF_PURCHASES']==0) & (Credit_card1['INSTALLMENTS_PURCHASES']==0):
        return 'none'
    if (Credit_card1['ONEOFF_PURCHASES']>0) & (Credit_card1['INSTALLMENTS_PURCHASES']>0):
         return 'both_oneoff_installment'
    if (Credit_card1['ONEOFF_PURCHASES']>0) & (Credit_card1['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (Credit_card1['ONEOFF_PURCHASES']==0) & (Credit_card1['INSTALLMENTS_PURCHASES']>0):
        return 'istallment'


# In[484]:

Credit_card1['purchase_type']=Credit_card1.apply(purchase,axis=1)


# In[485]:

Credit_card1.shape


# In[486]:

Credit_card1.head(5)


# In[487]:

Credit_card1['purchase_type'].value_counts()


# In[488]:

Credit_card1.groupby('purchase_type').apply(lambda x: np.mean(x['Monthly_cash_advance'])).plot.barh()

plt.title('Average cash advance taken by customers of different Purchase type : Both, None,Installment,One_Off')


# In[489]:

Credit_card1.groupby('purchase_type').apply(lambda x: np.mean(x['CREDIT_LIMIT'])).plot.barh()

plt.title('Credit limit of different Purchase types : Both, None,Installment,One_Off')


# In[490]:

Credit_card1.shape


# In[491]:

Credit_card1.dtypes


# In[492]:

Credit_card1.describe()


# In[493]:

#creating Dummies for categorical variable
Credit_card2=pd.concat([Credit_card1,pd.get_dummies(Credit_card1['purchase_type'])],axis=1)


# In[494]:

Credit_card2.head(3)


# In[495]:

print(Credit_card2.isnull().sum())


# In[496]:

#lets remove the redundent variable "purchase_type" as it is already conveyed using the dummy variables
x=['purchase_type','Percentage_oneoff_purchases','Percentage_Installment_purchase']
Credit_card3 = Credit_card2.drop(x, axis = 1) 


# In[497]:

print(Credit_card3.isnull().sum())


# In[498]:

Credit_card3.head(3)


# In[499]:

#Since there are variables having extreme values these can become potential outliers, hence lets standardize the data
#which will also bring all the variables in to one standard range
from sklearn import preprocessing

names = Credit_card3.columns

scaler = preprocessing.StandardScaler()

Credit_card4 = scaler.fit_transform(Credit_card3)
Credit_card4 = pd.DataFrame(Credit_card4, columns=names)


# In[500]:

Credit_card4.describe()


# In[501]:

Credit_card4.shape


# In[502]:

#lets apply PCA to reduse dimentionality before going for clustering the data, this also takes care of
#any correlation existing between the variables
from sklearn.decomposition import PCA
var_ratio={}
for n in range(2,25):
    pca=PCA(n_components=n)
    Credit_card5=pca.fit(Credit_card4)
    var_ratio[n]=sum(Credit_card5.explained_variance_ratio_)



# In[503]:

var_ratio


# In[504]:

pd.Series(var_ratio).plot()


# In[505]:

#we see that 13 components are explaining about 90% variance so we select 13 components


# In[530]:

pca_final=PCA(n_components=13)
Credit_card5=pca_final.fit_transform(Credit_card4)

Credit_card6=pd.DataFrame(Credit_card5)


# In[532]:

Credit_card6.shape


# In[533]:

Credit_card6.head(5)


# In[534]:

#lets name the principle components:

Principle_components = pd.DataFrame(pca_final.components_.T, columns=['PC_' +str(i) for i in range(13)],index=names)


# In[535]:

Principle_components.head(3)


# In[536]:

Principle_components.shape


# In[537]:

# Factor Analysis : variance explained by each principle component- 
pd.Series(pca_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(13)])


# In[538]:

#lets apply clustering algorithm to find the clusters those can explain customer behaviour and profile
#K-means clustering
#lets fine the optimumm no of clusters to be formed using silhouette_score method

from sklearn.cluster import KMeans

WSS = []
for i in range(1, 15):
  kmeans = KMeans(i)
  kmeans.fit(Credit_card6)
  WSS_iter = kmeans.inertia_
  WSS.append(WSS_iter)


# In[539]:

WSS


# In[540]:

number_clusters = range(1,15)
plt.plot(number_clusters,WSS)
plt.title('The Elbow Method')
plt.xlabel('number of clusters')
plt.ylabel('Within-sum of squares')


# In[541]:

#from the elbow graph we can see that K is equal to 6 should be optimum number of clusters


# In[542]:

#K-means clustering with K as 6:
km=KMeans(n_clusters=6,random_state=123) 
km.fit(Credit_card6)
km.labels_


# In[543]:

pd.Series(km.labels_).value_counts()


# In[544]:

#lets add the obtained clusters to the dataset
clustersPCA=pd.concat([Credit_card6, pd.DataFrame({'cluster':km.labels_})], axis=1)
clustersPCA.head(3)


# In[545]:

clusters=pd.concat([Credit_card1, pd.DataFrame({'cluster':km.labels_})], axis=1)
clusters.head(3)


# In[436]:

#lets plot the clusters


# In[549]:

x, y = Credit_card5[:, 0], Credit_card5[:, 1]

colors = {0:'red',1:'blue',2:'green',3:'yellow',4:'orange', 5:'purple'}

names = {0:'1', 1:'2', 2:'3', 3:'4', 4:'5',5:'6'}
    
plot = pd.DataFrame({'x': x, 'y':y, 'label':km.labels_}) 
groups = plot.groupby('label')

fig, ax = plt.subplots(figsize=(15, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
            color=colors[name],label=names[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.legend()
ax.set_title("Customers Segmentation based on their Credit Card usage behaviour.")
plt.show()


# In[445]:

#Evaluating k-meaans performance


# In[550]:

from sklearn.metrics import silhouette_score


# In[566]:

sil = [] 
kmax = 15

for k in range(2, kmax+1): 
    kmeans1 = KMeans(n_clusters = k).fit(Credit_card6) 
    labels = kmeans.labels_
    sil.append(silhouette_score(Credit_card6, labels, metric = 'euclidean'))


# In[567]:

number_clusters = range(1,15)
plt.plot(number_clusters,sil)
plt.title('The silhouette_score Method')
plt.xlabel('number of clusters - K')
plt.ylabel('silhouette_score')


# In[568]:

#Performance metrics also suggest that K-means with 6 cluster is able to show distinguished characteristics of each cluster.


# In[569]:

#cluster interpretation


# In[575]:

for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(plt.hist, c)


# In[ ]:

#Cluster0 People with average to high credit limit who make all type of purchases

#Cluster1 This group has more people with due payments who take advance cash more often

#Cluster2 Less money spenders with average to high credit limits who purchases mostly in installments

#Cluster3 People with high credit limit who take more cash in advance

#Cluster4 High spenders with high credit limit who make expensive purchases

#Cluster5 People who don't spend much money and who have average to high credit limit

