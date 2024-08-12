from ast import Import, ImportFrom
from operator import imod, index
from pyexpat import model
from symbol import import_as_name
from pandas.io import feather_format
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics

diabetes=datasets.load_diabetes()

#print(diabetes)

import pandas as pd
data=pd.DataFrame({
    "age":diabetes.data[ : , 0],
    "sex":diabetes.data[ : , 1],
    "bmi":diabetes.data[ :, 2],
    "bp":diabetes.data[ : , 3],
    "s1":diabetes.data[ : , 4],
    "s2":diabetes.data[ : , 5],
    "s3":diabetes.data[ : , 6],
    "s4":diabetes.data[ : , 7],
    "s5":diabetes.data[ : , 8],
    "s6":diabetes.data[ : , 9],
    "Disease pro":diabetes.target
    })

X=data[["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]]
Y=data[["Disease pro"]]

print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)
predict=model.predict(X_test)
print("----Predict")
print(predict)
print("----Real")
print(Y_test)

from sklearn import metrics
print("Accuracy is:",metrics.accuracy_score(predict,Y_test)*100)

print(model.predict([[1,3,5,7,2,8,4,9,0,6]]))


feather_imp=pd.Series(model.feature_importances_,index=diabetes.feature_names).sort_values(ascending=False)
print(feather_imp)

