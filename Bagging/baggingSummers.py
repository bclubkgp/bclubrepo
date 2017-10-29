# Bagged Decision Trees for Classification
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# Convert string column to integer

dataframe = pandas.read_csv('aircrash.csv', header=0)
dataframe['Age'].interpolate(method='akima')
del dataframe['Name']
dataframe.dropna(axis =0, how='any', inplace = True)
#dataframe.Name = pandas.to_numeric(dataframe.Name).fillna(0).astype(np.int64)

dataframe['Sex'].replace(['female','male'],[1,0],inplace=True)
dataframe['Embarked'].replace(['New York','Los Angeles','Chicago'],[0,1,2],inplace=True)
array = dataframe.values
#print array
X = array[:,2:10]
#print X
Y = array[:,1]
#print array[0]
#X.fillna(0)
seed = 7
kfold = model_selection.KFold(n_splits=9, random_state=seed)
cart = DecisionTreeClassifier()
kneighbors = KNeighborsClassifier();
num_trees = 75
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
for i in range(1, 500):
	model = BaggingClassifier(base_estimator=cart, n_estimators=i, random_state=seed)
	results = model_selection.cross_val_score(model, X, Y, cv=kfold)
	print(results.mean())
#model2 = BaggingClassifier(base_estimator=kneighbors, n_estimators=num_trees, random_state=seed)
#results2 = model_selection.cross_val_score(model, X, Y, cv=kfold)
#print(results2.mean())
