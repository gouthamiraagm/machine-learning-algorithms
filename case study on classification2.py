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

'''
exploring data analysis
1)getting to know the data
2)Data preprocessing(Missing values)
3)cross tables and data visualiztaion

'''
#to check variables data type
print(data.info())

#check for missing values
data.isnull()
print("Data columns with null values", data.isnull().sum())
#****No missing values!

#***summary of numerical varible
summary_num=data.describe()
print(summary_num)

#***summary of categorial varible
summary_cate=data.describe(include='O')
print(summary_cate)

#***frquency of each categories
#print(summary_cate) in this will get output as only 9 categories are there 
#to know what are there we will use value.counts()
data['JobType'].value_counts()
data['occupation'].value_counts()

#****for checking unique classes
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
#****there exicts '?' instead of nan

''''
go back to the data including null values  to read file

'''

data=pd.read_csv("income(1).csv",na_values=[' ?'])

#=========================================
#Data pre-processing
#==========================================
print(data.isnull().sum())

missing=data[data.isnull().any(axis=1)]
#axis=1==>to consider at least one columnvalue is missing

'''
points to note
1. missing values in Job type=1809
2. missing values in occupation=1816
3. there are 1809 rows where two specific columns i.e occupation &jobtype  having missing values
4. (1816-1809)=7==> you still have occupation unfilled for these 7 rows .because job type is never worked

'''

data2=data.dropna(axis=0)
#we dont konw waht fill in these cases so lets drop all incomplete values

# ====================================================
# LOGISTIC REGRESSION
# ====================================================
#machine learning directly will not work on categorical data
#so we need to convert the data into 0 and 1
#reindexing salary status names 0 to 1
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
new_data=pd.get_dummies(data2,drop_first=True) #mapping string values to integer values

#storing columns list
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

#confusion matrix
confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

#calculating accuracy
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

#Printing the misclassified values from prediction
print("Missclassified samples :%d" %(test_y !=prediction).sum())



