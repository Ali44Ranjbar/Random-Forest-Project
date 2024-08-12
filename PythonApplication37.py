import imp
from operator import index
from pyexpat import features, model
from sklearn.model_selection import train_test_split
from sklearn import datasets


breastcancer=datasets.load_breast_cancer()
#print(breastcancer)
import pandas as pd
data=pd.DataFrame({
    "radius":breastcancer.data[ : , 0],
    "texture":breastcancer.data[ : , 1],
    "perimeter":breastcancer.data[ :, 2],
    "area":breastcancer.data[ : , 3],
    "smoothness":breastcancer.data[ : , 4],
    "compactness":breastcancer.data[ : , 5],
    "concavity":breastcancer.data[ : , 6],
    "concave points":breastcancer.data[ : , 7],
    "symmetry":breastcancer.data[ : , 8],
    "fractal dimenson":breastcancer.data[ : , 9],
    "type":breastcancer.target
    })
X=data[["radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimenson"]]
Y=data[["type"]]
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train , X_test ,Y_train , Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)
p=model.predict(X_test)
print("-------Predict-------")
print(p)
print("--------Real------")
print(Y_test)

from sklearn import metrics 
print("Accuracy is :",metrics.accuracy_score(p , Y_test)*100)

print(model.predict([[0,4,2,6,8,1,3,5,7,9]]))

print(model.feature_importances_)
print(breastcancer.feature_names)

feature_imp = pd.Series(model.feature_importances_ , index= breastcancer.feature_names).sort_values(ascending = False)
print(feature_imp)



