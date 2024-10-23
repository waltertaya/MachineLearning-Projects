'''
1. Design model (input, output size, forward pass)
2. Construct loss and optimizer
3. Training loop
    - forward pass: compute prediction
    - backward pass: gradients
    - update weights
'''

import torch
import torch.nn as nn

X = torch.tensor([[2], [3], [5], [7]], dtype=torch.float32)
y = torch.tensor([[6], [9], [15], [21]], dtype=torch.float32)

X_test = torch.tensor([[9], [10], [1], [0]], dtype=torch.float32)

# n_samples, n_features = X.shape

# input_size = n_features
# output_size = n_samples

input_size = 1
output_size = 1

model = nn.Linear(input_size, output_size)

learning_rate = 0.01
epochs = 900

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    y_pred = model(X)

    l = loss(y, y_pred)

    l.backward() # dl/dw

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 200 == 0:
        [w, b] = model.parameters()
        print(f'Epoch {epoch}: weight = {w[0][0].item():.2f} loss = {l:.5f}')

# print(f'Predict x = 9, y = {model(X_test).item():.2f}')
y_test_pred = model(X_test)

predicted_values = y_test_pred.squeeze().tolist()

print(f'Predict for x = [9, 10, 1, 0], y = {[f"{val:.2f}" for val in predicted_values]}')

