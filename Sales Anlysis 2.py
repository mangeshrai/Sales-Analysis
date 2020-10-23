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

from itertools import combinations
from collections import Counter

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 25)
pd.set_option("display.max_rows", 100)


alldata=pd.read_csv('alldatacleaned.csv')
print(alldata.head())

def get_city(address):                                                                                  #makes function
    return address.split(',')[1]                                                                        #returns City spilt by commas

def get_state(address):
    return address.split(',')[2].split(' ')[1]                                                           #returns state

alldata['City']=alldata['Address'].apply(lambda x: get_city(x) + ' ' + get_state(x))
print(alldata.head())


alldata['City']=alldata['Address'].apply(lambda x: get_city(x) + ' (' + get_state(x) + ')')
print(alldata.head())


print(alldata.groupby('City').sum())                                                   ##Monthly sales Summary


G1=alldata.groupby('City').SalesValue.sum().plot(kind='bar', figsize=(15,4))
plt.ylabel('Sales in USD')
plt.xlabel('City')
# plt.show(G1)


alldata['OrderDate']=pd.to_datetime(alldata['OrderDate'])                               #changes Orderdate column to date time format

alldata['Hour'] = alldata['OrderDate'].dt.hour
alldata['Minute'] = alldata['OrderDate'].dt.minute
print(alldata.head())

print(alldata.groupby('Hour').Hour.count())

G2=alldata.groupby('Hour').Hour.count().plot(kind='line', figsize=(15,4))
plt.xticks()
plt.grid()
plt.ylabel('Orders')
plt.xlabel('Hour')
plt.show(G2)

##what products are sold together most often

df=alldata[alldata['OrderID'].duplicated(keep=False)]                                   #shows duplicates of OrderID
print(df.head(20))


df['Grouped']=df.groupby('OrderID')['Product'].transform(lambda x: ','.join(x))          #joins Product column by commas
print(df.head())

df=df[['OrderID','Grouped']].drop_duplicates()                                            #drops duplicate OrderID
print(df.head(20))



# print(df['Grouped'].value_counts())

# df.to_csv('alldatagrouped.csv', index=False)


# out=df['Grouped'].str.split(',\s+', expand=True).stack().value_counts()
# print(out)


count = Counter()
for row in df['Grouped']:
    row_list=row.split(',')
    count.update(Counter(combinations(row_list, 3)))

for key, value in count.most_common(10):
     print(key, value)


#What product sold the most


productgroup=alldata.groupby('Product')                                                                 #groupbs by product to give total by products
quantity=productgroup.sum()['Quantity']

Product=[Product for Product, df in productgroup]

g4=plt.bar(Product, quantity)
plt.xticks(rotation='vertical', size=8)
plt.ylabel('Quantity')
plt.xlabel('Product')
# plt.show(g4)

prices=alldata.groupby('Product').mean()['Price']                                                       #gives mean prices of products
print(prices)

fig, ax1=plt.subplots()

ax2=ax1.twinx()
ax1.bar(Product, quantity, color='g')
ax2.plot(Product, prices, 'b-')

ax1.set_xlabel('Product')
ax1.set_ylabel('Quantity', color='g')
ax2.set_ylabel('Price', color='b')
ax1.set_xticklabels(Product, rotation='vertical')
plt.show(ax2)


