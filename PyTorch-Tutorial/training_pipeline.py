import torch
import torch.nn as nn

X = torch.tensor([2, 3, 5, 7], dtype=torch.float32)
y = torch.tensor([6, 9, 15, 21], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forwardpass(x):
    return x * w

learning_rate = 0.01

epochs = 25

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(1, epochs):
    y_pred = forwardpass(X)

    l = loss(y, y_pred)

    l.backward() # dl/dw

    optimizer.step()

    optimizer.zero_grad()

    w.grad.zero_()

    if epoch % 5 == 0:
        print(f'Epoch {epoch}: weight = {w:.2f} loss = {l:.5f}')

print(f'Predict x = 9, y = {forwardpass(9):.2f}')
