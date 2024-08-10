import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data=pd.read_csv('DigitalAd_dataset.csv')
X=data.iloc[:,:-1].values
Y=data.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=40)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(f"Accuracy of the model :{accuracy_score(y_test,y_pred)*100}%")

age=int(input("Enter the customer age "))
sal=int(input("Enter the customer salary "))
newCust=[[age,sal]]
result=model.predict(sc.transform(newCust))
print(result)
if result==1:
    print("Customer will buy ")
else:
    print("Customer will not buy")
