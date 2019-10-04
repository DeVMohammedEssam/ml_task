
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Google_Stock_Price_Train.csv")


training_set = dataset.iloc[:,1:2].values
dataset_n_rows = training_set.shape[0]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
scaled_training_set = sc.fit_transform(training_set)

x_train =[]
y_train = []

for i in range(60,dataset_n_rows):
    x_train.append(scaled_training_set[i-60:i,0])
    y_train.append(scaled_training_set[i,0])

x_train = np.array(x_train) 
y_train = np.array(y_train) 
n_predictors = 1


x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))    
    
import tensorflow as tf

TPU_CONFIG = tf.contrib.tpu.TPUConfig(num_cores_per_replica=4)
run_config_class=tf.contrib.tpu.RunConfig()
run_config_class.__init__(tpu_config=TPU_CONFIG)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()
no_of_lstm_layers=50
#return sequences is used when using multiple LSTMs
dropout_ratio = 0.2 #used to regularize the NN and prevent overfitting
#1st LSTM
regressor.add(LSTM(units=no_of_lstm_layers,return_sequences = True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(dropout_ratio))
#2nd LSTM
regressor.add(LSTM(units=no_of_lstm_layers,return_sequences = True))
regressor.add(Dropout(dropout_ratio))
#3rd LSTM
regressor.add(LSTM(units=no_of_lstm_layers,return_sequences = True))
regressor.add(Dropout(dropout_ratio))

#4th LSTM
regressor.add(LSTM(units=no_of_lstm_layers)) #return sequence is false as we won't add another LSTM
regressor.add(Dropout(dropout_ratio))
#output layer
regressor.add(Dense(units=1,activation='sigmoid'))

#we choosed loss= 'mean squared error' as it is a regression problem
regressor.compile(optimizer='adam' , loss ='mean_squared_error')

regressor.fit(x_train,y_train,epochs=20,batch_size = 32)


dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
stock_prices = dataset.iloc[:,1:2].values
dataset_total = pd.concat((dataset['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test=[]

for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
    
x_test = np.array(x_train) 
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))    
 
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(stock_prices,color='red', label='real stock prices')
plt.plot(predicted_stock_price,color='blue', label='predicted stock prices')

plt.title('Google stcok prices predictions')
plt.xlabel('time')
plt.ylabel('Google stock price')
plt.legend()
plt.show()