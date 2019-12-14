import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense



# convert an array of values into a dataset matrix
def form_matrix(dataset, window_size=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-window_size-1):
        a = dataset[i:(i+window_size), 0]
        dataX.append(a)
        dataY.append(dataset[i + window_size, 0])
    return np.array(dataX), np.array(dataY)



mean_temperature = pd.read_csv('temperature_data.csv', usecols=[2,3], engine='python')
mean_temperature['YEARMODA'] = pd.to_datetime(mean_temperature['YEARMODA'], format='%Y%m%d')
print(mean_temperature)


# plt.plot(mean_temperature['YEARMODA'],mean_temperature['TEMP'])
# plt.show()

temperature_vals = mean_temperature['TEMP'].values
temperature_vals = temperature_vals.astype('float32')

"""split the data into training and teststing sets"""

train_size = int(len(temperature_vals) * 0.69)
test_size = len(temperature_vals) - train_size
train, test = temperature_vals[0:train_size,:], temperature_vals[train_size:len(temperature_vals),:]

"""reshape dataset to a matrix format and use a 2 day window for t-1 & t for predicting t+1"""
window_size = 2
trainX, trainY = form_matrix(train, window_size)
testX, testY = form_matrix(test, window_size)


"""create and fit Multilayer Perceptron model"""
model = Sequential()
model.add(Dense(12, input_dim=window_size, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# shift train predictions for plotting
trainPredictPlot = np.empty_like(temperature_vals)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[window_size:len(trainPredict)+window_size, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(temperature_vals)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(window_size*2)+1:len(temperature_vals)-1, :] = testPredict
# plot baseline and predictions
plt.plot(temperature_vals)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()