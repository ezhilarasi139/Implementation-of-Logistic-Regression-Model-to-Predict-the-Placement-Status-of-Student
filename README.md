# EX NO 5 : Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect and preprocess data and scale numerical features

2.import labelencoder and fit transform data

3.import train test split and logistic regression to train features

4.import and print the classification report
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: EZHILARASI N 
RegisterNumber:  212224040088
*/

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis=1)#removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") #A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
<img width="1441" height="268" alt="image" src="https://github.com/user-attachments/assets/6ea64fae-8d32-4a8a-8d93-2b9b482f045e" />

<img width="1272" height="252" alt="image" src="https://github.com/user-attachments/assets/167dfca8-fd1f-49de-9761-8fda602fb273" />

<img width="267" height="382" alt="image" src="https://github.com/user-attachments/assets/31c98df2-0691-4c02-92b4-b0f93480c39c" />

<img width="453" height="43" alt="image" src="https://github.com/user-attachments/assets/f8b0a932-cce8-43aa-85ef-c4d1ddde6198" />

<img width="1182" height="562" alt="image" src="https://github.com/user-attachments/assets/5d976643-ac61-474b-82c1-4d39ec57008b" />

<img width="1105" height="563" alt="image" src="https://github.com/user-attachments/assets/7e6ab18a-54e0-440d-8efc-6c4acf15d5ad" />

<img width="552" height="348" alt="image" src="https://github.com/user-attachments/assets/bdfdedeb-ca98-441f-b501-247ea0dbc28f" />

<img width="898" height="87" alt="image" src="https://github.com/user-attachments/assets/1c83c1e7-0322-44c1-b7e7-765842f64d7d" />

<img width="529" height="81" alt="image" src="https://github.com/user-attachments/assets/717686e3-6d06-4309-89eb-0d1060b342da" />

<img width="705" height="245" alt="image" src="https://github.com/user-attachments/assets/66f143d8-4247-46a3-bb61-ab7baefb825c" />

<img width="451" height="43" alt="image" src="https://github.com/user-attachments/assets/541bd74b-5368-4bff-aa8c-e38dad20a214" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
