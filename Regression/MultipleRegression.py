import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


# Cost Computing
def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    x = x_train
    y = np.array(y_train)
    total_cost = 0
    for i in range(m-5):
        z = np.array(x.iloc[i:i+5])
        f_wb = np.dot(w, z) + b
        total_cost += (f_wb - y[i-1])**2
    total_cost /= 2 * (m-4)
    return total_cost


# Computing Cost Gradient
def compute_gradient(x_train, y_train, w, b):
    m = x_train.shape[0]
    n = w.shape[0]
    x = np.array(x_train.iloc[:])
    y = np.array(y_train)
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m-5):
        err = np.dot(x[i:i+5], w) + b - y[i-1]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * x[i+j]

        dj_db += err
    dj_dw = dj_dw / (m-4)
    dj_db = dj_db / (m-4)

    return dj_dw, dj_db


# Computing the gradient descent
def gradient_descent(x_train, y_train, w_in, b_in, alpha_):
    w = w_in
    b = b_in
    j = []
    iterations = 10000
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w, b)

        w = (w - alpha_ * dj_dw)
        b -= alpha_ * dj_db

        if i < iterations:
            z = (compute_cost(x_train, y_train, w, b))
            j.append(z)

        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j[-1]:8.2f}",
                  f"w1: {w[0]: 0.3e}, w2: {w[1]: 0.3e}, w3: {w[2]: 0.3e}, w4: {w[3]: 0.3e}, w5: {w[4]: 0.3e},",
                  f" b:{b: 0.5e}")

    return w, b


# Predicting output using train and test data
def predict(x_train, y_train, x_test, y_test, w_in, b_in, alpha_):
    w, b = gradient_descent(x_train, y_train, w_in, b_in, alpha_)
    m = x_test.shape[0]
    yhat = np.zeros(y_test.shape[0])
    for i in range(m-5):
        yhat[i] = np.dot(x_test[i:i+5], w) + b
    return yhat


# Reading the data
data = pd.read_csv('')  # Here goes csv filename inside quotes
prediction = np.zeros(data.shape[0])
new_df = pd.DataFrame(columns=['Closing', 'Prediction'])
new_df['Closing'] = data['Close'].values
z = data['Date'].tolist()
data = data.drop("Close", axis="columns")

# Train-test data splitting
X_train = new_df.iloc[0:round(new_df.shape[0] * 0.7), 0]
X_test = new_df.iloc[round(new_df.shape[0]*0.7):, 0]
ytrain = new_df.iloc[5:round(new_df.shape[0] * 0.7) + 5, 0]
ytest = new_df.iloc[round(new_df.shape[0]*0.7)+5:, 0]

# Defining the values of parameters and alpha (slope coefficient)
print("")
w1 = np.zeros(5)
b1 = 0
alpha = 1e-10

y_pred = predict(X_train, ytrain, X_test, ytest, w1, b1, alpha)

# Calculating Error
mse = round(mean_squared_error(ytest, y_pred), 2)
rmse = round(math.sqrt(mean_squared_error(ytest, y_pred)), 2)
map = round(mean_absolute_percentage_error(ytest, y_pred), 2)
r2 = round(r2_score(ytest, y_pred), 7)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAP: {map}')
print(f'R2: {r2}')

# Graphical Representation of the result obtained
t = 82 #Change the t value according size of dataset
n = round(data.shape[0] * 0.7)
date = z[n: :t]
plt.plot(np.array(ytest), c='b', label="Original")
plt.plot(y_pred, c='r', label="Predicted")
plt.title('Multiple Regression Model')
plt.xticks(np.arange(0, round(new_df.shape[0] * 0.3), t), date)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper right')
plt.show()

