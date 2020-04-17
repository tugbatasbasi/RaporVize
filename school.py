import pandas as pd #Gerekli Kütüphaneleri import ettik
import numpy as np
df=pd.read_csv("student-mat.csv")
print(df.info())

x=df.drop(['G3'],axis=1)
x=pd.concat([x,pd.get_dummies(x)],axis=1)
x=x.select_dtypes(exclude=['object'])

x = x.loc[:,~x.columns.duplicated()] # kopyaları sildik
y=df['G3'].values

print(x.info())


##%% normalization
from sklearn.preprocessing import Normalizer



scaler = Normalizer()
X = scaler.fit_transform(x)

#
#X=x



#%% Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

linear_reg = LinearRegression()
linear_reg.fit(X, y)



ypred_linear = linear_reg.predict(X)

r2_lin=r2_score(y,ypred_linear)
linear_mse = mean_squared_error(y, ypred_linear)
linear_mae = mean_absolute_error(y, ypred_linear)



print("Linear Accuracy: ",r2_lin)
print("Linear MSE: ",linear_mse)
print("Linear MAE: ",linear_mae)
#%% Decision Tree
from sklearn.tree import DecisionTreeRegressor

decisiontree_reg=DecisionTreeRegressor()
decisiontree_reg.fit(X,y)



ypred_decision=decisiontree_reg.predict(X)



r2_dt=r2_score(y,ypred_decision)
decision_mse = mean_squared_error(y, ypred_decision)
decision_mae = mean_absolute_error(y, ypred_decision)



print("DecisionTree Accuracy: ",r2_dt)
print("DecisionTree MSE: ",decision_mse)
print("DecisionTree MAE: ",decision_mae)

#%% RanfomForest Regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=42)


randomforest_reg=RandomForestRegressor(n_estimators=100, random_state=42)
randomforest_reg.fit(x_train,y_train)


ypred_rf=randomforest_reg.predict(x_test)




r2_rf=r2_score(y_test,ypred_rf)
rf_mse = mean_squared_error(y_test, ypred_rf)
rf_mae = mean_absolute_error(y_test, ypred_rf)



print("RanfomForest Accuracy: ",r2_rf)
print("RanfomForest MSE: ",rf_mse)
print("RanfomForest MAE: ",rf_mae)
