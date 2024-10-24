import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# Data preparation
X_np, y_np = datasets.make_regression(n_samples=200, n_features=4, random_state=True, noise=14)
X = torch.from_numpy(X_np.astype('float32'))
y = torch.from_numpy(y_np.astype('float32'))

y = y.view(200, 1)

# Model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.lin(x)

model = LinearRegression(4, 1)

learning_rate = 0.01
epochs = 500

# optimizer and loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Iterations
for epoch in range(1, epochs + 1):
    # forward pass and loss
    y_pred = model(X)
    l = loss(y_pred, y)

    # backward pass
    l.backward()

    # updates w, b
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'===== epoch {epoch} l: {l:.4f} =======')

# Predictions
y_predicted = model(X).detach().numpy()

# Plotting (use only the first feature for visualization)
plt.scatter(X_np[:, 0], y_np, color='red', label='Original data')
plt.scatter(X_np[:, 0], y_predicted, color='blue', label='Fitted line')
plt.legend()
plt.show()
