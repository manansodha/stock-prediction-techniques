from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# Reading the data
data = pd.read_csv('daily_data_n50_1.csv') # Here goes csv filename inside quotes
target = np.array(data['Close'].values)
data = data.drop("Close", axis="columns")
data_ = data[["High", "Low"]]
z = data['Date'].tolist()
data = data.drop("Date", axis="columns")


# Train-test data split
data_train = data_.iloc[0:round(data_.shape[0]*0.7), :]
data_test = data_.iloc[round(data_.shape[0]*0.7):, :]
target_train = target[0:round(target.shape[0]*0.7)]
target_test = target[round(target.shape[0]*0.7):]

map = []
mse = []
rmse = []
r2 = []
# Cycling through number of nearest neighbours to find the one with least error
for i in range(1, 20, 2):
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(data_train, target_train)
    target_pred = knn.predict(data_test)

    # Calculating error values for nearest neighbour values in the loop to find the optimal one
    map.append(round(mean_absolute_percentage_error(target_test, target_pred), 2))
    mse.append(round(mean_squared_error(target_test, target_pred), 2))
    rmse.append(round(math.sqrt(mean_squared_error(target_test, target_pred)), 2))
    r2.append(round(r2_score(target_test, target_pred), 7))



df = pd.DataFrame({'Neighbours': range(1, 20, 2), 'map': map, 'mse': mse, 'rmse': rmse, 'r2':r2})
df = df.sort_values(by=['map', 'mse', 'rmse', 'r2'])
print(df)

# Predicting using the optimal number of nearest neighbours
knn = KNeighborsRegressor(n_neighbors=df.head().iloc[0,0])
knn.fit(data_train, target_train)
target_pred = knn.predict(data_test)


# Graphical Comparision of acutal vs predicted data
t = 82 #Change the t value according size of dataset
n = round(data.shape[0] * 0.7)
date = z[n: :t]
plt.plot(target_test, color='b', label='Original')
plt.plot(target_pred, color='r', label='Predicted')
plt.legend(loc='upper right')
plt.xticks(np.arange(0, round(data.shape[0] * 0.3), t), date)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('KNN Regresssion Model')
plt.show()
