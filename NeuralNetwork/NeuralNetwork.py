import numpy as np
"""
dot()返回的是两个数组的点积(dot product)
1.如果处理的是一维数组，则得到的是两数组的內积
2.如果是二维数组（矩阵）之间的运算，则得到的是矩阵积
"""


# 双曲正切函数
def tanh(x):
    return np.tanh(x)


# 双曲正切函数的导数
def tanh_derivative(x):
    return 1 - pow(tanh(x), 2)


# 标准逻辑函数
def logistic(x):
    return 1 / (1 + np.exp(-x))


# 标准逻辑函数的导数
def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used.
        Can be "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_derivative = logistic_derivative
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative

        self.weights = []

        # 初始化输入层和隐含层之间的权值[-0.25 - 0.25]
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i-1] + 1, layers[i] + 1))-1)*0.25)  # ????????????????????????????????????????????????
        # 初始化输出层权值[-0.25 - 0.25]
        self.weights.append((2 * np.random.random((layers[i] + 1, layers[i+1]))-1)*0.25)  # ????????????????????????????????????????????????

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        # 保证所有的输入X至少是二维数组,如果是一维数组则会转化为一个二位的1*len(X)的数组
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])  # 初始化一个全为1的矩阵 列数+1是为了有B这个偏向
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]  # 从所有训练样本中随机选一组

            for l in range(len(self.weights)):  # going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l])))  # Computer the node value for each layer (O_i) using activation function

            # 反向递推计算delta:从输出层开始,先算出该层的delta,再向前计算
            error = y[i] - a[-1]  # Computer the error at the top layer
            deltas = [error * self.activation_derivative(a[-1])]  # For output layer, Err calculation (delta is updated error)

            # Staring backprobagation
            # 从倒数第2层开始反向计算delta
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                # Compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_derivative(a[l]))
            deltas.reverse()  # 逆转列表中的元素

            # 逐层调整权值
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        a = np.concatenate((np.array(x), np.ones(1)))
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
