import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Data preparation
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
n_samples, n_features = X.shape

# Scale
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

X_train = torch.from_numpy(X_train.astype('float32'))
X_test = torch.from_numpy(X_test.astype('float32'))
y_train = torch.from_numpy(y_train.astype('float32'))
y_test = torch.from_numpy(y_test.astype('float32'))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid((self.lin(x)))

model = LogisticRegression(n_features)

learning_rate = 0.01
epochs = 1000

# optimizer and loss
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Iterations
for epoch in range(1, epochs + 1):
    # forward pass and loss
    y_pred = model(X_train)
    l = loss(y_pred, y_train)

    # backward pass
    l.backward()

    # updates w, b
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'===== epoch {epoch} l: {l:.4f} =======')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {accuracy:.5f}')
