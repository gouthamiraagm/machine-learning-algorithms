import pandas as pd #to work with data frames
import numpy as np  # to perform numerical opreations
import seaborn as sns  #to visualize data
import os

#to partition the data
from sklearn.model_selection import  train_test_split

#importing library  for logistic regression
from sklearn.linear_model import LogisticRegression

#importing performance metrics -accuracy  score &confusion
from sklearn.metrics import accuracy_score,confusion_matrix


os.chdir("C:\\Users\\Gouthami Ragam\\Desktop\\pandas")
data_income=pd.read_csv("income(1).csv")
data=data_income.copy()

data=pd.read_csv("income(1).csv",na_values=[' ?'])
data2=data.dropna(axis=0)

# ==================================================================
# LOGISTIC REGRESSION-- REMOVING INSIGNIFIACNT VARIBLES
# =====================================================================

data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

cols=['gender','nativecountry','race','JobType']
new_data=data2.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)

#storing columns names
columns_list=list(new_data.columns)
print(columns_list)

#separating the inputs names from the data
features=list(set(columns_list)-set(['SalStat']))#we are excluding the salary stauts
print(features)

#storing the output values in y
y=new_data['SalStat'].values
print(y)

#storing the  values from input  features
x=new_data[features].values
print(x)

#splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#random_state=0(if you set this one same set of random samples will be choosen on every run)


#make an instance of model
logistic=LogisticRegression()

#Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

#prediction from the test data
prediction=logistic.predict(test_x)
print(prediction)

#calculating accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

#Printing the misclassified values from prediction
print("Missclassified samples :%d" %(test_y !=prediction).sum())


#================================================================
#KNN
#===================================================================

#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#importing library for plotting
import matplotlib.pyplot as plt

#storing the K nearest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

#fitting the values for x and y
KNN_classifier.fit(train_x,train_y)

#predicting the test values with model
prediction=KNN_classifier.predict(test_x)

#performance matrix check
confusion_matrix=confusion_matrix(test_y,prediction)
print("\t","predicted values")
print("original values","\n",confusion_matrix)

#calculating accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)


print("Missclassified samples :%d" %(test_y !=prediction).sum())


missclassified_sample=[]
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)#train_x=input model,trian_y=output model
    pred_i=knn.predict(test_x)
    missclassified_sample.append((test_y !=pred_i).sum())
print(missclassified_sample)
    