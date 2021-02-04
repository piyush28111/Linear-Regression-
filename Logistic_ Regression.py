
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

url = "https://stats.idre.ucla.edu/stat/data/binary.csv"
df= pd.read_csv(url)
df

df.head()

x= df[['gre','gpa']]
y= df['admit']

x
y

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1)
xtrain
xtest
ytrain
ytest

lg= LogisticRegression().fit(xtrain,ytrain)

ypred = lg.predict(xtest)
ypred

metrics.accuracy_score(ytest,ypred)
metrics.precision_score(ypred,ytest)
metrics.recall_score(ytest, ypred)

metrics.classification_report(ytest, ypred)
mat= metrics.confusion_matrix(ytest, ypred)
sns.heatmap(mat, annot=True,cbar=True)




confusion_matrix = pd.crosstab(ytest,ypred,rownames=['Actual'],colnames=['Predicted'])
confusion_matrix
accuracy = (19+2)/(19+2+15+4)
accuracy

new_candidates = {'gre':[670,438,555,799,619],'gpa':[3.4,4.3,2.3,4.4,3.5]}
x_sample = pd.DataFrame(new_candidates)
x_sample
y_pred_sample = lg.predict(x_sample)
y_pred_sample

pd.concat([x_sample,pd.Series(y_pred_sample,name='Admit')],axis=1)


## log2 


url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/pima-indians-diabetes.csv'
df= pd.read_csv(url,header=None,names=['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label'])

df.columns
x= df[['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']]
x
y= df['label']
y

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.1)
ytest

lg = LogisticRegression().fit(xtrain,ytrain)
lg

ypred= lg.predict(xtest)
ypred

lg.score(xtrain,ytrain)

metrics.accuracy_score(ypred,ytest)
metrics.precision_score(ypred,ytest)
metrics.recall_score(ypred,ytest)
from sklearn.metrics import classification_report
print(classification_report(ypred,ytest))

confusion_mt = metrics.confusion_matrix(ytest,ypred)
confusion_mt
(50+11)/(50+11+4+12)
sns.heatmap(pd.DataFrame(confusion_mt),annot=True,cbar=True)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

ypred_pro = lg.predict_proba(xtest)
ypred_pro
pd.DataFrame(ypred_pro)
ypred_prob= ypred_pro[::,1]
ypred_prob

ytest
fpr,tpr,_= metrics.roc_curve(ytest,ypred_prob)
tpr
fpr

auc = metrics.roc_auc_score(ytest,ypred_prob)
auc

plt.plot(fpr,tpr,label='auc = '+str(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

