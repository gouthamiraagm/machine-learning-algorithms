import pandas as pd #to work with data frames
import numpy as np  # to perform numerical opreations
import seaborn as sns  #to visualize data
import os

os.chdir("C:\\Users\\Gouthami Ragam\\Desktop\\pandas")
cars_data=pd.read_csv("cars_sampled.csv")
cars=cars_data.copy()
cars.info()
cars.describe() #summarizing data
pd.set_option('display.float_format',lambda x:'%.3f' % x) #after point only 3 value we will get
cars.describe()
pd.set_option('display.max_columns',500) #dispaly maximun five columns
cars.describe()

#==============================================================
#Dropping unanted columns
#==============================================================

col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)

#==============================================================
#removing duplicate records
#==============================================================

cars.drop_duplicates(keep='first',inplace=True)
#470 duplicate records

#==============================================================
#data cleaning
#no of missing value in each column
#==============================================================
cars.isnull().sum()

#variable  year of registration
yearwise_count=cars['yearOfRegistration' ].value_counts().sort_index()
sum(cars['yearOfRegistration' ]>2018)#output=26
sum(cars['yearOfRegistration' ]<1950)#output=38
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=cars) #scatterplot
#working range 1950 to 2018


#variable price
price_count=cars['price' ].value_counts().sort_index()
sns.distplot(cars['price' ])
cars['price' ].describe()
sns.boxplot(y=cars['price'])
sum(cars['price' ]>150000)#output=34
sum(cars['price' ]<100)#output=1748
#working range 100 to 150000


#variable powerPS
price_count=cars['powerPS' ].value_counts().sort_index()
sns.distplot(cars['powerPS' ])
cars['powerPS' ].describe()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)
sum(cars['powerPS' ]>500)#output=115
sum(cars['powerPS' ]<10)#output=5565
#working range 10 to 500

#==============================================================
#working range of data
#==============================================================
cars=cars[
        (cars.yearOfRegistration<=2018)
        &(cars.yearOfRegistration>=1950)
        &(cars.price>=100)#lowerbound
        &(cars.price<=150000)#upperbound
        &(cars.powerPS>=10)
        &(cars.powerPS<=500)]

#6700 records are dropped

#further to simplify varible reduction
#combining year Of Registration and month Of Registration

#creating a new varible Age bay adding yearOfRegistration and monthOfRegistration
cars['monthOfRegistration' ]/=12
cars['Age']=(2018-cars['yearOfRegistration' ])+cars['monthOfRegistration' ]
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

#==============================================================
#Dropping yearOfRegistration and monthOfRegistration
#==============================================================

cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#visualizing the parameter

#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])


#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

##visualizing the parameters after narrowing working range
#Age vs Price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=cars)#scatter plot

#powerPS vs Price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=cars)#scatter plot
#as power ps increases price also  increases


#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#fewer cars have commercial==>insignificant

#variable  offer type
cars['offerType'].value_counts()
sns.countplot(x='offerType',data=cars)
#all cars have offers==>insignificant

#varible abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
#equally distributed
sns.boxplot(x='abtest',y='price',data=cars)
#for every price value there is almost 50-50 distribution
#does not affect price-Insignificant


#varible vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)
#vechile type  affect price

#varible gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
# gearbox affect price

#varible model
cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)
#cars are distributed ovet many models
#considered in modelling


#varible kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.countplot(x='kilometer',data=cars)
sns.boxplot(x='kilometer',y='price',data=cars)
cars['kilometer'].describe()
#considered in modelling

#varible fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#fuelType effcts the price

#varible brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)

sns.boxplot(x='brand',y='price',data=cars)
#cars are distributed ovet many brands
#considered in modelling

#varible notRepairedDamage
#yes===car is damaged but not rectified
#no====car was damaged but has been rectified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)


#==============================================================
#Removing insiginifacnt varible
#==============================================================
col=['seller','offerType','abtest']
cars=cars.drop(columns=col,axis=1)
cars_copy=cars.copy()

#=====================================================================
#COrrelation
#======================================================================
cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation,3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]

#=============================================================================

'''
we are going to build a Linear regreesion  and Random Forest Model on two sets of data
1.Data obtained by omitting rows with any missing values
2.data obtained by imputing the missing values
'''
#=====================================================================
#OMITTING MISSING VALUES
#======================================================================
cars_omit=cars.dropna(axis=0)#row

#converting categorial varible to dummy variable
cars_omit=pd.get_dummies(cars_omit,drop_first=True)

#=====================================================================
#importing necessary libraries
#======================================================================
#to partition the data
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#=====================================================================
#Model buliding with omitted data
#======================================================================
x1=cars_omit.drop(['price'],axis='columns',inplace=False)
y1=cars_omit['price']

#plotting the varibel vs price
prices=pd.DataFrame({'1.before':y1,'2.after':np.log(y1)})
prices.hist()

#transforming the price as logarithmic value
y1=np.log(y1)

#splitting the data into test and train
X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
#test_size=0.3 menas 70% of the data will go to the train set and 30% of data will go the test set
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


#=====================================================================
#base line model for omitted data
#======================================================================
'''
we are making a base model by using a test data mean value
this is set to the benchmark and to compare with our regression model
'''
#finding the mean of the tested value
base_pred=np.mean(y_test)
print(base_pred)

#repeating same value till length of test data
base_pred=np.repeat(base_pred,len(y_test))

#finding the RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test,base_pred))
print(base_root_mean_square_error)







