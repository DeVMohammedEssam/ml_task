#data preprocessing start
import pandas as pd
import numpy as np


dataSet = pd.read_csv("Churn_Modelling.csv")
X = dataSet.iloc[:,3:13].values
Y = dataSet.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
LabelEncoder_x_1 = LabelEncoder() #for country
LabelEncoder_x_2 = LabelEncoder() #for gender

X[:,1]=LabelEncoder_x_1.fit_transform(X[:,1])
X[:,2]=LabelEncoder_x_2.fit_transform(X[:,2])


#make variables dummy and avoid dummy trap
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#columnt transformer applies transformers on data just like oneHotEncoder
ColumnTransformer_x = ColumnTransformer(
            [('one_hot_encoder',OneHotEncoder(categories='auto'),[1])],
            remainder = 'passthrough'
        )
X = np.array(ColumnTransformer_x.fit_transform(X) , dtype = np.float64)
X = X[:,1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state = 0 )


from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
#data preprocessing end

#ANN building
import keras
from keras.models import Sequential
from keras.layers import Dense


def build_classifier():
    
    classifier = Sequential()
    classifier.add(Dense(output_dim=6 , kernel_initializer='uniform',activation='relu',input_dim=11))#first hidden layer
    classifier.add(Dense(output_dim=6 , kernel_initializer='uniform',activation='relu')) #second hidden layer
    classifier.add(Dense(output_dim=1 , kernel_initializer='uniform',activation='sigmoid',input_dim=11)) #output layer
    classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])
    return classifier

#withoud k-fold cross validation

'''
classifier = build_classifier()
classifier.fit(x_train,y_train,batch_size=20,epochs=200)

y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5) #make all y_pred above 0.5 == true

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)


#predict a new observation
newObserv = np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])
newObserv = sc_x.transform(newObserv)
newPredection = classifier.predict(newObserv)
newPredection = (newPredection>0.5)
'''

#uisng k corss vaidation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

classifier = KerasClassifier(build_fn=build_classifier , batch_size = 10 , epochs = 100 )
accuracies = cross_val_score(estimator = classifier, X=x_train , y= y_train, cv=10 , n_jobs=-1 )

mean = accuracies.mean()
variance = accuracies.std()


#ANN tuning
""" from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCv


classifier = KerasClassifier(build_fn=build_classifier , batch_size = 10 , epochs = 100 )
accuracies = cross_val_score(estimator = classifier, X=x_train , y= y_train, cv=10 , n_jobs=-1 )

mean = accuracies.mean()
variance = accuracies.std() """
















