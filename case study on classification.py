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

#realationship between independent varibles
correlation=data2.corr()
#correlation values are lies in between [-1 to+1] here in the data none of the values are near to 1
#all  are near to 0 so non of them are corrleated
#if they are near to 1 then we can say that there is strong relationship between the two variables


#==================================
#Cross tables and data visualization 
#===================================
#extract the column names
data2.columns


#======================================================
#Gender proportion table
#=======================================================
gender=pd.crosstab(index=data2["gender"],columns='count',normalize=True)

print(gender)

#===========================================================
#Gender vs Salary status
#==============================================================
gender_salstat=pd.crosstab(index=data2["gender"],
                           columns=data2["SalStat"],
                           margins=True,
                           normalize="index")
print(gender_salstat)

#=============================================================
#Frequency Distribution of "Salary status"
#=============================================================
SalStat=sns.countplot(data2["SalStat"])

'''75% of people's salary is <50000
   25% of people's salary is >50000
'''
############Histogram of Age################
sns.distplot(data2['age'],bins=10,kde=False)
#people with age 20-45 age are high in frequency

###########box plot age vs salary status###############
sns.boxplot('SalStat','age',data=data2)
data2.groupby('SalStat')['age'].median()


#barplot for JobType vs Salary Status
sns.countplot(y='JobType',hue='SalStat',data=data2)

#education vs Salary Status
sns.countplot(y='EdType',hue='SalStat',data=data2) #bar plot

EdType_salstat=pd.crosstab(index=data2["EdType"],
                           columns=data2["SalStat"],
                           margins=True,
                           normalize="index")
print(EdType_salstat)




