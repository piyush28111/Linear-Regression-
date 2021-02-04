from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

data=pd.read_csv('dat.csv')
data

x=data.X.values.reshape(-1,1)
x.shape
y=data.Y.values.reshape(-1,1)
y.shape

x,y
plt.scatter(x,y)
plt.plot(x,y)

lr= LinearRegression()

xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3)
xtest
ytest
plt.scatter(xtest,ytest)
plt.plot(xtest,ytest,marker='o')
plt.grid(True)


xtest
ytest

lr.fit(xtrain,ytrain)
y_pred=lr.predict(xtest)
y_pred

plt.scatter(xtrain,ytrain)
#plt.scatter(xtest,ytest)
plt.scatter(xtest,y_pred)

r_sq= lr.score(xtrain,ytrain)
r_sq



#################

data= pd.read_csv('iris.csv')
data
data.columns
data.head()

x= data[['sepal_length','sepal_width','petal_length','petal_width']]
x
y= data['name']
y

y[y=='setosa']='1'
y[y=='versicolor']='2'
y[y=='virginica']='3'

y

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)

xtrain
xtest
ytrain
ytest

xtrain.shape,xtest.shape,ytrain.shape,ytest.shape

lr=LinearRegression().fit(xtrain,ytrain)

lr.score(xtrain,ytrain)



xtest 
ytest
ypred= lr.predict(xtest)
y_pred=ypred.round()
y_pred



# General example 

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from pydataset import data 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

mtcars= data('mtcars')
mtcars.columns 
mtcars.columns 
data= mtcars[['mpg','wt','hp']]
data
data.head()

from statsmodels.formula.api import ols 

modelset = ols('mpg ~ wt + hp',data=data).fit()
modelset.summary()

#modelset2 = ols('mpg ~ wt ',data=data).fit()
#modelset2.summary()

#modelset3 = ols('mpg ~  hp',data=data).fit()
#modelset3.summary()
model_pred = modelset.predict()
model_pred


#fig, ax = plt.subplots(figsize=(12, 8))
#fig = sm.graphics.plot_ccpr(modelset, "wt", ax=ax)

plt.scatter(data['wt'],data['mpg'])

f=plt.figure(figsize=(12, 8))
sm.graphics.plot_ccpr_grid(modelset,fig=f)

