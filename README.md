# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program: 
```
Program to implement the SVM For Spam Mail Detection..
Developed by: S Dhanush
RegisterNumber:  25005353
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
*/
```
## Output:
![SVM For Spam Mail Detection](sam.png) 
## DATA
<img width="876" height="513" alt="Screenshot 2025-11-12 112223" src="https://github.com/user-attachments/assets/b88ec0be-1e46-4d80-81a9-c59d8fb9f214" /> 

## 
<img width="94" height="52" alt="Screenshot 2025-11-12 111835" src="https://github.com/user-attachments/assets/2d8a3dd3-0578-4095-9210-452b9781103c" /> 

## 
<img width="87" height="39" alt="Screenshot 2025-11-12 111846" src="https://github.com/user-attachments/assets/d20115ab-7bab-48ad-9a26-1feb79902177" />

## 
<img width="1560" height="204" alt="Screenshot 2025-11-12 111914" src="https://github.com/user-attachments/assets/e679ce28-08b4-4945-ab5e-049c73e38289" />

## 
<img width="672" height="44" alt="Screenshot 2025-11-12 111942" src="https://github.com/user-attachments/assets/a8bcce32-77a9-40c7-888e-5f0ebaec9949" />

## 
<img width="193" height="34" alt="Screenshot 2025-11-12 112009" src="https://github.com/user-attachments/assets/bb90c7c6-6ae2-48f1-a21e-548c054ceb07" />

##
<img width="148" height="58" alt="Screenshot 2025-11-12 112019" src="https://github.com/user-attachments/assets/d219231b-23eb-4c8c-b026-03be08779358" /> 

## 
<img width="635" height="212" alt="Screenshot 2025-11-12 112030" src="https://github.com/user-attachments/assets/3d1111a1-0acb-4472-a695-a5e4f8c3426e" />




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
