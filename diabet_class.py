import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("original.csv")
df=df.rename(columns={"Outcome": "Case"})#Outcome ismi case olarak değiştirildi



print(df.info())

y = df['Case'].values
X=df.drop(["Case"],axis=1)

#%% Normalization
x=(X-np.min(X))/(np.max(X)-np.min(X))



#x=X


#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
knn=KNeighborsClassifier(n_neighbors=7)#16,7
knn.fit(x_train,y_train)
knn_prediction=knn.predict(x_test)
f1_knn=f1_score(y_test,knn_prediction)
hamming_knn=hamming_loss(y_test,knn_prediction)

print("knn accury: ",knn.score(x_test,y_test))
print("knn f1 score: ",f1_knn)
print("knn hamming loss: ",hamming_knn)


# hyper parameter
#%% find best k value
score_list=[]

for each in range(1,120):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,120),score_list)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

#%% SVM
from sklearn.svm import SVC
svm=SVC(random_state=1,gamma="auto")
svm.fit(x_train,y_train)
svm_prediction=svm.predict(x_test)

f1_svm=f1_score(y_test, svm_prediction, average='weighted', labels=np.unique(svm_prediction))

#f1_svm=f1_score(y_test,svm_prediction)
hamming_svm=hamming_loss(y_test,svm_prediction)

print("svm accuracy: ",svm.score(x_test,y_test))
print("svm f1 score: ",f1_svm)
print("svm hamming loss: ",hamming_svm)
#%% DT Classification

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_prediction=dt.predict(x_test)
f1_dt=f1_score(y_test,dt_prediction)
hamming_dt=hamming_loss(y_test,dt_prediction)
print("dt accuracy: ",dt.score(x_test,y_test))
print("dt f1 score: ",f1_dt)
print("dt hamming loss: ",hamming_dt)
#%% RF Classification

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
rf_prediction=rf.predict(x_test)
f1_rf=f1_score(y_test,rf_prediction)
hamming_rf=hamming_loss(y_test,rf_prediction)
print("rf accuracy: ",rf.score(x_test,y_test))
print("rf f1 score: ",f1_rf)
print("rf hamming loss: ",hamming_rf)





