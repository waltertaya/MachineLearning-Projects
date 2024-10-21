import numpy as np

np.random.seed(0)

# Generate random data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

import pandas as pd

df = pd.DataFrame(X, columns=['X'])
df['y'] = y

df.to_csv('data.csv', index=False)

w = np.random.rand(2, 1)
print(w)

w += 0.1

print(w)
