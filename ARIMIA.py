import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rpy2.robjects.packages import importr
graphics = importr('graphics')
grdevices = importr('grDevices')
base = importr('base')
stats = importr('stats')
forecast = importr('forecast')

from statsmodels.tsa.seasonal import seasonal_decompose


mean_temperature = pd.read_csv('temperature_data.csv', usecols=[2,3], engine='python')
mean_temperature['YEARMODA'] = pd.to_datetime(mean_temperature['YEARMODA'], format='%Y%m%d')
mean_temperature.set_index('YEARMODA', inplace=True)

#print(mean_temperature)

# result = seasonal_decompose(mean_temperature, model='multiplicative')
# result.plot()
# plt.show()
