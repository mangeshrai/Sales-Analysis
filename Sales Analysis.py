import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import scale
from collections import Counter
from scipy.interpolate import *
from scipy.stats import *
import os


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 25)
pd.set_option("display.max_rows", 100)


#Merging 12 months of sales data into single files

df=pd.read_csv('./Sales Data/Sales_April_2019.csv')                                   #reads April Sales file in Sales Data folder
print(df.head())

files=[file for file in os.listdir('./Sales Data')]                                     #list all CSV files in Sales data folder

alldata=pd.DataFrame()

for file in files:                                                                          #concats files
    df = pd.read_csv('./Sales Data/' + file)
    alldata = pd.concat([alldata, df])

print(alldata.head())

# alldata.to_csv('alldata.csv', index=False)                                                  #makes new CSV file with all csv data


alldata=pd.read_csv('alldata.csv')
print(alldata)


alldata.columns=['OrderID','Product','Quantity','Price','OrderDate', 'Address']                 #changes column names
print(alldata.head())


##Dropping NAN values

nandf=alldata[alldata.isna().any(axis=1)]                                                        #will display nan values
print(nandf)
alldata=alldata.dropna(how='all')                                                                  #drops NAN values
print(alldata)

tempdf=alldata[alldata['OrderDate'].str[0:5]=='Order']                                        #shows rows that contain Or
print(tempdf)

alldata=alldata[alldata['OrderDate'].str[0:5]!='Order']                                       #drops rows that contain Or
print(alldata)

##Adding Month, Week, and Weekday column

alldata['OrderDate']=pd.to_datetime(alldata['OrderDate'], errors='coerce')                                         ##converts Date column to date time
alldata['Month']=alldata.OrderDate.dt.month                                                         #adds column month from OrderDate
print(alldata.head())
alldata['Week']=alldata.OrderDate.dt.week                                                                    #adds column of what week number according to date column
print(alldata.head())
alldata['WeekNumber']=alldata.OrderDate.dt.weekday_name                                  #adds weekly number column
print(alldata.head())

# alldata.to_csv('alldatacleaned.csv', index=False)



##Adding Sales Value column

alldata['Quantity']=pd.to_numeric(alldata['Quantity'])                                  #changes column to numeric values
alldata['Price']=pd.to_numeric(alldata['Price'])
print(alldata.head())

alldata['SalesValue']=alldata['Quantity']*alldata['Price']                                #adds column Sales Value, error because
print(alldata.head())

alldata.to_csv('alldatacleaned.csv', index=False)

##Monthly Sales analysis

print(alldata.groupby('Month').sum())                                                   ##Monthly sales Summary

G1=alldata.groupby('Month').SalesValue.sum().plot(kind='bar', figsize=(15,4))
plt.ylabel('Sales in USD')
plt.xlabel('Month')
plt.show(G1)

G1=alldata.groupby('Month').SalesValue.sum().plot(kind='bar', figsize=(15,4))


print(alldata.groupby('Week').sum())
