import numpy as np


a = np.array([[1, 2, 3]])
b = np.array([[1, 2, 3]])
print(a.dot(b.T))
print(np.dot(a.T, b))


