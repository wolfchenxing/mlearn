import numpy as np
import math
import matplotlib.pylab as plt


def computeCorrelation(X, Y):
    """
    计算变量X和Y的相关系数
    """
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0  # 分子
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += diffXXBar * diffYYBar
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST


# 利用np.ployfit求拟合系数，并求出其决定系数
def polyfit(x, y, degree):
    result = {}
    # 多项式拟合(从给定的x,y中解析出最接近数据的方程式)。degree为多项式最高次幂，结果为多项式的各个系数
    coffs = np.polyfit(x, y, degree)
    result['polynomial'] = coffs.tolist()
    p = np.poly1d(coffs)  # #将系数代入方程，得到函式p
    yhat = p(x)  # 估计的y值
    plt.plot(x, yhat, 'ob-', label='prediction points')  # 绘制预测点和线段
    ybar = np.mean(y)
    x1 = range(11)
    y1 = [ybar for i in x1]
    plt.plot(x1, y1, color='g', label='average of y')  # 绘制y的平均值
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    result['determination'] = ssreg / sstot
    return result


testX = [1, 3, 8, 7, 9]
testY = [10, 12, 24, 21, 34]

correlation = computeCorrelation(testX, testY)

print("相关系数：", correlation)
print("决定系数：", correlation ** 2)

plt.xlim([0, 10])
plt.ylim([0, 40])

plt.scatter(testX, testY, color='black', label="sample points")

print(polyfit(testX, testY, 1))
plt.legend()  # 使图例生效
plt.show()
