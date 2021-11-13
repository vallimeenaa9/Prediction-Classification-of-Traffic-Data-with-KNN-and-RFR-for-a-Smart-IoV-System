import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Book1.csv')
x=dataset.iloc[:,12:15].values
y=dataset.iloc[:,3:4].values


#encoding dependant variable
from sklearn.preprocessing import LabelEncoder #import label encoder class from sklearn lib preprocessing module
le=LabelEncoder() #create object
y=le.fit_transform(y)#fit transform

y = y.reshape(-1, 1)
##from sklearn.compose import ColumnTransformer #import column transformer class from sklearn lib compose module
from sklearn.preprocessing import OneHotEncoder#import one hot encoder func from sklearn lib pre-processing modulke
one_hot_encoder = OneHotEncoder(categorical_features = [0])
##ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough') #create object with arguments
##y=np.array(ct.fit_transform(y)).toarray()#convert into numpy array an then fit trnsform
y = one_hot_encoder.fit_transform(y).toarray()

#split test train set
from sklearn.model_selection import train_test_split #import train test split func  from sklearn lib model selection module
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.1,random_state=0) #provide arguments


#fitting the rfr model
from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=100 ,random_state=0)#n_estimators stand for the number of trees
rfr.fit(x_train,y_train)

#predicting rfr result
y_pred=rfr.predict([[14,4,3]])


    

