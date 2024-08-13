import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
# Load the stock market data
data = pd.read_csv("") # CSV filename goes in the quotes

# Extract features and target
X = data[["Open", "High", "Low"]]
y = data["Close"]
z = data['Date'].tolist()

# Split data into training and testing sets
X_train = X[0:round(X.shape[0] * 0.7)]
X_test = X[round(X.shape[0] * 0.7):]
y_train = np.array(y[0:round(X.shape[0] * 0.7)])
y_test = np.array(y[round(X.shape[0] * 0.7):])

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the MLP regressor
mlp = MLPRegressor(
   hidden_layer_sizes=(100, 50), # Layer 1 with 100 nodes and Layer 2 with 50 nodes
   activation="identity", # Linear activation
   solver="lbfgs",  # Adam optimizer
   alpha=0.0001,  # Regularization strength
   max_iter=2000,  # Maximum iterations
   random_state=42,  # Ensure reproducibility
)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = mlp.predict(X_test)

# Evaluate model performance
map = round(mean_absolute_percentage_error(y_test, y_pred), 7)
print(f"MAPE: {map}")
mse = round(mean_squared_error(y_test, y_pred), 7)
print(f"MSE: {mse}")
rmse = round(math.sqrt(mean_squared_error(y_test, y_pred)), 7)
print(f"RMSE: {rmse}")
r2 = round(r2_score(y_test, y_pred), 7)
print(f'R2: {r2}')

n = round(data.shape[0] * 0.7)
t = 15 #number of ticks on x-axis = size(y_test) / t
date = z[n: :t]
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Original', c='b')
plt.plot(y_pred, label='Predicted', c='r')
plt.xticks(np.arange(0, round(data.shape[0] * 0.3), t), date)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Multi-layer Perceptron Regression Model')
plt.legend(loc='upper right')
plt.show()

