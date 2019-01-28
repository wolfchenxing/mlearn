import numpy as np
import random


# m denotes the number of examples here, not the number of features
def gradientDescent(x, y, theta, alpha, m, numIterations):
    """
    利用梯度下降算法求参数theta
    :param x: 输入实例
    :param y: 分类标签
    :param theta: 要学习的参数
    :param alpha: 学习率
    :param m: 实例个数
    :param numIterations: 迭代次数
    :return: 学习好的参数theta
    """
    xTrans = x.transpose()  # 矩阵的转置
    for i in range(0, numIterations):
        hypothsis = np.dot(x, theta)  # 估计值
        loss = hypothsis - y
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d / Cost: %f" % (i, cost))
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient  # 更新法则
    return theta


def genData(numPoints, bias, variance):
    """
    创建测试数据
    :param numPoints: 实例数
    :param bias: 偏向
    :param variance: 方差
    :return: 测试实例及其分类标签x,y
    """
    x = np.zeros(shape=(numPoints, 2))  # 初始化numPoints行2列(x1,x2)的全零元素矩阵
    y = np.zeros(shape=numPoints)  # 归类标签
    for i in range(0, numPoints):
        x[i][0] = 1  # 所有行第1列为：1
        x[i][1] = i  # 所有行第2列为：行的数目
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
m, n = np.shape(x)
numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)
