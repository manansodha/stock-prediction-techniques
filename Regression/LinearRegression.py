import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import math

# Computing cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0
    for k in range(m):
        f_wb = w * (x[k]) + b
        # cost = (f_wb - y[k]) ** 2
        total_cost += (f_wb - y[k]) ** 2
    total_cost /= (2 * m)
    return total_cost


# Computing the cost gradient
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * (x[i]) + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


# Computing the descent of the cost gradient
def compute_descent(x, y, w_in, b_in, alpha_):
    j = []
    p = []
    w = w_in
    b = b_in
    it = 10000
    # Adjusting the values of parameters using alpha
    for i in range(it):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha_*dj_dw
        b -= alpha_*dj_db
        if i < it:
            j.append(compute_cost(x, y, w, b))
            p.append([w, b])
        if i % math.ceil(it/10) == 0:
            print(f"Iteration {i:4}: Cost {j[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b


# Predicting using the training and testing data
def predict(x_train, y_train_, x_test, y_test_, w_in, b_in, alpha_):
    w, b = compute_descent(x_train, y_train_, w_in, b_in, alpha_)
    m = x_test.shape[0]
    yhat = np.zeros(m)
    for i in range(m):
        yhat[i] = (w*x_test[i] + b)

    return yhat

# Reading the data
data = pd.read_csv('') # Here goes csv filename inside quotes
Y = np.array(data['Close'].values)
data = data.drop("Close", axis="columns")

# Creating the target data
X = np.zeros(data.shape[0])
for f in range(data.shape[0]):
    # Input for training and testing is mean of highest and lowest price of the day
    X[f] = (data["High"][f] + data["Low"][f]) / 2

# Train-test data split
X_train = X[0:round(X.shape[0] * 0.7)]
X_test = X[round(X.shape[0] * 0.7):]
y_train = np.array(Y[0:round(X.shape[0] * 0.7)])
y_test = np.array(Y[round(X.shape[0] * 0.7):])

# Defining the values of parameters and alpha (slope coefficient)
print("")
w1 = 0
b1 = 0
alpha = 1e-8

y_pred = predict(X_train, y_train, X_test, y_test, w1, b1, alpha)

#Calculating Error
mse = round(mean_squared_error(y_test, y_pred), 2)
rmse = round(math.sqrt(mean_squared_error(y_test, y_pred)), 2)
map = round(mean_absolute_percentage_error(y_test, y_pred), 2)
r2 = round(r2_score(y_test, y_pred), 7)
print(f'MAP: {map}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAP: {r2}')
# Graphical representation
t = 82 #Change the t value according size of dataset
z = data['Date'].tolist()
date = z[round(X.shape[0] * 0.7): :t]
plt.plot(y_test, label='Original')
plt.plot(pd.Series(y_pred), c='r', label='Predicted')
plt.xticks(np.arange(0, round(X.shape[0] * 0.3), t), date)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Linear Regression Model')
plt.legend(loc="upper right")
plt.show()
