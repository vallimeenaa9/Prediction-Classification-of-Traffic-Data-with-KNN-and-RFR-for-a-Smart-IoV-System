import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('book.csv')
x=dataset.iloc[:,12:15].values
y=dataset.iloc[:,3:4].values


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() 
y=le.fit_transform(y)

y = y.reshape(-1, 1)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("STREET", OneHotEncoder(), [0])], remainder = 'passthrough')
y = ct.fit_transform(y).toarray()
##from sklearn.preprocessing import OneHotEncoder
##one_hot_encoder = OneHotEncoder(categorical_features = [0])
##ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough') #create object with arguments
##y=np.array(ct.fit_transform(y)).toarray()#convert into numpy array an then fit trnsform
##y = one_hot_encoder.fit_transform(y).toarray()


from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.1,random_state=0) 


y_train= y_train.astype(np.int64)
y_test= y_test.astype(np.int64)
from sklearn.ensemble import RandomForestRegressor
rfr= RandomForestRegressor(n_estimators=100 ,random_state=0)
rfr.fit(x_train,y_train)


y_pred=rfr.predict([[14,4,3]])



#checking the intensity of conjestion

road=dataset.iloc[:,3:4]

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder() 
road=le.fit_transform(road)

road = road.reshape(-1, 1)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Country column
ct = ColumnTransformer([("ROAD", OneHotEncoder(), [0])], remainder = 'passthrough')
road = ct.fit_transform(road).toarray()
#one_hot_encoder = OneHotEncoder(categorical_features = [0])
#road = one_hot_encoder.fit_transform(road).toarray()

x_speed=dataset.iloc[:,[12,13,14]].values
x_speed=np.append(road,x_speed,axis=1)
y_speed=dataset.iloc[:,2:3].values

for i in y_speed:
    i[i<10]=0
    i[i>=10]=1
pd.DataFrame(y_pred)
    
        
#split test train set
from sklearn.model_selection import train_test_split #import train test split func  from sklearn lib model selection module
x_speed_train,x_speed_test,y_speed_train,y_speed_test =train_test_split(x_speed,y_speed,test_size=0.25,random_state=0) #provide arguments

y_speed_train= y_speed_train.astype(np.float64)
y_speed_test= y_speed_test.astype(np.float64)
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=25,metric='minkowski',p=2)
classifier.fit(x_speed_train,y_speed_train)
accuracy = classifier.score(x_speed_test, y_speed_test)
y_speed_pred=classifier.predict(x_speed_test)

y_test= y_test.astype(np.float64)
from sklearn.metrics import confusion_matrix
confusionmatrix=confusion_matrix(y_speed_test,y_speed_pred)

arr = np.array(y_pred)
xcoor=np.arange(0,25)
ycoor=y_pred
xcoor=xcoor.reshape(-1,1)
ycoor=ycoor.reshape(-1,1)
plt.plot(xcoor,ycoor,'ob')





