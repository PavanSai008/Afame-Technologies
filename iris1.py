#import libraries


import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 


#columns = ["sepel length" , 'sepel width' , 'petel length' , 'petel width', 'specie']

#loadind the dataset

file_path=("iris.csv")
df=pd.read_csv(file_path)
df.head()   #to display first 5 entries

df.describe()   # give a statstical description
sns.pairplot(df,hue='species')     # gives plot


# seperating input and output columns

data = df.values

x = data[:,0:4]
y = data[:,4]
print(x)
print(y)


#spliting data for testing and training


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)  
#train is the columns given for trainig and test for testing in this case as we gave 0.2 size we get 120 train and 30 test
print(x_train) #or any [up]

# SUPPORT VECTOR MACHINE ALGORITHM

from sklearn.svm import SVC

model_svc =SVC()
model_svc.fit(x_train,y_train)

#model is trained
predition1 = model_svc.predict(x_train)
predition2 = model_svc.predict(x_test)
#calc accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,predition1)) 
print(accuracy_score(y_test,predition2))

