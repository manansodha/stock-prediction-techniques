import math
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Reading the data
data = pd.read_csv('')  # Here goes csv filename inside quotes
z = data['Date'].tolist()
data = data.drop("Date", axis="columns")
Y = np.asarray(data['Close'].values).astype(np.float32)
X = data.drop(['Close'], axis="columns")


# Create lag features for data
def create_lagged_features(data, lag):
    X_lagged = []
    for i in range(len(data) - lag):
        X_lagged.append(data[i:i+lag])
    return np.array(X_lagged)


lag = 5  # Number of lagged features
X_lagged = create_lagged_features(Y, lag)
Y = Y[lag:]  # Target values

# Train-test data splitting
X_train = X_lagged[0:round(X.shape[0] * 0.7)]
X_test = X_lagged[round(X.shape[0] * 0.7):]
y_train = np.asarray(Y[0:round(X.shape[0] * 0.7)]).astype(np.float32)
y_test = np.asarray(Y[round(X.shape[0] * 0.7):]).astype(np.float32)

# Training the data
svm_regressor = SVR(kernel='linear', C=1.0)
svm_regressor.fit(X_train, y_train)

# Predicting output using the given data
y_pred = svm_regressor.predict(X_test)

# Calculating the error
mse = round(mean_squared_error(y_test, y_pred), 2)
rmse = round(math.sqrt(mean_squared_error(y_test, y_pred)), 2)
map = round(mean_absolute_percentage_error(y_test, y_pred), 2)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAP: {map}')

# Graphical comparision of actual vs predicted data
t = 0 #Change the t value according size of dataset
n = round(data.shape[0] * 0.7)
date = z[n: :t]
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Original', c='b')
plt.plot(y_pred, label='Predicted', c='r')
plt.xticks(np.arange(0, round(data.shape[0] * 0.3), t), date)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regressor Model')
plt.legend(loc='upper right')
plt.show()
