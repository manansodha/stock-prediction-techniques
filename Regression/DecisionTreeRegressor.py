import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Reading the data
data = pd.read_csv('') # Here goes csv filename inside quotes
target = data['Close'].values
z = data['Date'].tolist()
data = data.drop(['Close', 'Date'], axis='columns')
data_train = data.iloc[0:round(data.shape[0]*0.7), :]
data_test = data.iloc[round(data.shape[0]*0.7):, :]
target_train = target[0:round(target.shape[0]*0.7)]
target_test = target[round(target.shape[0]*0.7):]

# Cycling through depth values to find the one with least error
map = []
mse = []
rmse = []
for i in range(10, 200):
    reg = DecisionTreeRegressor(max_depth=i)
    reg.fit(data_train, target_train)

    target_pred = reg.predict(data_test)

    # Calculating error values for each depth in the loop to find the optimal one
    map.append(round(mean_absolute_percentage_error(target_test, target_pred), 2))
    mse.append(round(mean_squared_error(target_test, target_pred), 2))
    rmse.append(round(math.sqrt(mean_squared_error(target_test, target_pred)), 2))


df = pd.DataFrame({'depth': range(10, 200), 'map': map, 'mse': mse, 'rmse': rmse})
df = df.sort_values(by=['map', 'mse', 'rmse'])
print(df)

# Predicting using the optimal depth value
reg = DecisionTreeRegressor(max_depth=df.head().iloc[0,0])
reg.fit(data_train, target_train)

target_pred = reg.predict(data_test)

# Graphical Comparision of acutal vs predicted data
t = 0 #Change the t value according size of dataset
n = round(data.shape[0] * 0.7)
date = z[n: :t]
plt.plot(target_test, color='b', label='Original')
plt.plot(target_pred, color='r', label='Predicted')
plt.title('Decision Tree Regression model')
plt.xticks(np.arange(0, round(data.shape[0] * 0.3), t), date)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper right')
plt.show()
