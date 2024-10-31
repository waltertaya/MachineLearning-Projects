# Steps
'''
1. predict(X, w, b) => { return np.matmul(X, w) + b }
2. loss(y, y_pred) => { return (1/n) * np.sum((y - y_pred)**2)} # MSE
3. linear_regression_step(): # Loop
    iterate through the epochs
    updates the w and b => { w = w - alpha * (2/n) * np.sum(y - y_pred) }
'''
import pandas as pd
from sgd import sgd
from bgd import bgd
from loss import loss
from predict import predict

learning_rate = 0.0001
epochs = 10000

df = pd.read_csv('data.csv')
y = df['y'].to_numpy()
X = df.drop('y', axis=1).to_numpy()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         

# print(f'y: {y}')
# print(f'X: {X}')

# SGD
print('SGD')
w, b = sgd(X, y, learning_rate, epochs)

df = pd.read_csv('test.csv')
y_test = df['y'].to_numpy()
X_test = df.drop('y', axis=1).to_numpy()

y_pred = predict(X_test, w, b)

print(f'y test: {y_test}')
print(f'y pred: {y_pred}')

l = loss(y_test, y_pred)

print(f'Loss: {l:.4f}')

# BGD
print('BGD')
w, b = bgd(X, y, learning_rate, epochs)

df = pd.read_csv('test.csv')
y_test = df['y'].to_numpy()
X_test = df.drop('y', axis=1).to_numpy()

y_pred = predict(X_test, w, b)

print(f'y test: {y_test}')
print(f'y pred: {y_pred}')

l = loss(y_test, y_pred)

print(f'Loss: {l:.4f}')
