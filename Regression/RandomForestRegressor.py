from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Reading data and splitting to train and test
data = pd.read_csv('') # Here goes csv filename inside quotes
target = data['Close'].values
z = data['Date'].tolist()
data = data.drop(['Close', 'Date'], axis='columns')
data_train = data.iloc[0:round(data.shape[0] * 0.7), :]
data_test = data.iloc[round(data.shape[0] * 0.7):, :]
target_train = target[0:round(target.shape[0]*0.7)]
target_test = target[round(target.shape[0]*0.7):]

map = []
mse = []
rmse = []
# Cycling through depth values to find the one with least error
for i in range(10, 201, 10):
    reg = RandomForestRegressor(max_depth=i, random_state=10)
    reg.fit(data_train, target_train)

    target_pred = reg.predict(data_test)

    # Calculating error values for each depth in the loop to find the optimal one
    map.append(round(mean_absolute_percentage_error(target_test, target_pred), 2))
    mse.append(round(mean_squared_error(target_test, target_pred), 2))
    rmse.append(round(math.sqrt(mean_squared_error(target_test, target_pred)), 2))
    print(f'Done {i}')

df = pd.DataFrame({'depth': range(10, 201, 10), 'map': map, 'mse': mse, 'rmse': rmse})
df = df.sort_values(by=['map', 'mse', 'rmse'])
print(df)

# Predicting using the optimal depth value
reg = RandomForestRegressor(max_depth=df.head().iloc[0,0], random_state=10)
reg.fit(data_train, target_train)

target_pred = reg.predict(data_test)

# Graphical Comparision of acutal vs predicted data
t = 0 #Change the t value according size of dataset
n = round(data.shape[0] * 0.7)
date = z[n: :t]
plt.plot(target_test, color='b', label='Original')
plt.plot(target_pred, color='r', label='Predicted')
plt.xticks(np.arange(0, round(data.shape[0] * 0.3), t), date)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Random Forest Regressor Model')
plt.legend(loc='upper right')
plt.show()

