#Random Forest:Consisting of decision trees

####Finding decision tree with higher accuracy than others:
from pyexpat import model
from sklearn.model_selection import train_test_split 
from sklearn import datasets
####Finding and downloading dataset:
iris=datasets.load_iris()
print(iris)
#print(iris.data)
#print(iris.target_names) 


####Naming the columns given in the output:
import pandas as pd
data=pd.DataFrame({
    "sepal length":iris.data[ : , 0],
    "sepal width":iris.data[ : , 1],
    "petal length":iris.data[ :, 2],
    "petal width":iris.data[ : , 3],
    "type":iris.target
    })
X=data[["sepal length","sepal width","petal length","petal width"]] #features=X
Y=data[["type"]] #target=Y
print(X)
print(Y)
####Testing the data using 30% of data:
from sklearn.model_selection import train_test_split 
X_train , X_test ,Y_train , Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)  ####Find the model           ####model: decision tree that is more accurate than others
model.fit(X_train,Y_train)
p=model.predict(X_test)
print("-----Predict")
print(p)
print("----Real")
print(Y_test)

####Eran accuracy to percent:
from sklearn import metrics
print("Accuracy is :",metrics.accuracy_score( p , Y_test)*100)      

print(model.predict([[2,3,4,5]]))

###### Data Normalization/ Data Cleaning
##### Feature Importances

feature_imp = pd.Series(model.feature_importances_ , index= iris.feature_names).sort_values(ascending = False)
print(feature_imp)

import matplotlib as plt
import seaborn as sb

sb.barplot(x=feature_imp , y = feature_imp.index)
plt.show()





