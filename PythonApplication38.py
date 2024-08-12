from gettext import install
from pyexpat import model
from sklearn.model_selection import train_test_split
from sklearn import datasets

w=datasets.load_wine()

#print(w)
import pandas as pd
data=pd.DataFrame({
    "alcohol":w.data[ : , 0],
    "malic acid":w.data[ : , 1],
    "ash":w.data[ : , 2],
    "alcalinity of ash":w.data[ : , 3],
    "magnesium":w.data[ : , 4],
    "total phenols":w.data[ : , 5],
    "flavanoids":w.data[ : , 6],
    "nonflavanoid phenols":w.data[ : , 7],
    "proanthocyanins":w.data[ : , 8],
    "color intensity":w.data[ : , 9],
    "hue":w.data[ : , 10],
    "od280/od315 of diluted wines":w.data[ : , 11],
    "proline":w.data[ : , 12],
    "class":w.target
    })
X=data[["alcohol","malic acid","ash","alcalinity of ash","magnesium","total phenols","flavanoids","nonflavanoid phenols","proanthocyanins","color intensity","hue","od280/od315 of diluted wines", "proline"]]
Y=data[["class"]]
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)
p=model.predict(X_test)
print("------Predict------")
print(p)
print("------Real-----")
print(Y_test)

from sklearn import metrics
print("Accuracy is:",metrics.accuracy_score(p , Y_test)*100)

print(model.predict([[1,3,5,7,9,11,2,4,6,8,10,0,12]]))

