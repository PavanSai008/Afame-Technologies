#import libraries


import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
#%matplotlib inline 


#columns = ["sepel length" , 'sepel width' , 'petel length' , 'petel width', 'specie']

#loadind the dataset
file_path="iris.csv"
df=pd.read_csv(file_path)
print(df.head(5))   #to display first 5 entries

print(df.describe())   # give a statstical description
sns.pairplot(df,hue='species')    # gives plot
plt.show()


# seperating input and output columns
print("the seperated input and output")
data = df.values

x = data[:,0:4]
y = data[:,4]
print(x) #sepal_length  sepal_width  petal_length  petal_width
print(y) #species


#spliting data for testing and training


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)  
#train is the columns given for trainig and test for testing in this case as we gave 0.2 size we get 120 train and 30 test
#print(x_train) #or any [up] to see the data given for traing and testing

# SUPPORT VECTOR MACHINE ALGORITHM

from sklearn.svm import SVC

model_svc =SVC()
model_svc.fit(x_train,y_train)

#model is trained
predition1 = model_svc.predict(x_train)
predition2 = model_svc.predict(x_test)
#calc accuracy
from sklearn.metrics import accuracy_score
print("the accuracy of training set",accuracy_score(y_train,predition1)) 
print("the accuracy of testing set",accuracy_score(y_test,predition2))
