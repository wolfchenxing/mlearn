from NeuralNetwork import NeuralNetwork
import numpy as np


XORArray = [[0, 0], [0, 1], [1, 0], [1, 1]]

nn = NeuralNetwork([2, 2, 1], 'tanh')
X = np.array(XORArray)
y = np.array([0, 1, 1, 0])
nn.fit(X, y)
for i in range(len(XORArray)):
    print("样本：%s，属于：%s，预测：%s" % (XORArray[i], y[i], nn.predict(XORArray[i])))


