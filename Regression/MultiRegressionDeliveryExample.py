import numpy as np
from sklearn import linear_model

"""
一家快递公司送货：X1：运输里程   X2： 运输次数   Y：总运输时间
"""
deliveryData = np.genfromtxt("deliveryData.csv", delimiter=",")
print("deliveryData:", "\n", deliveryData)

# 最后一列为因变量，区分自变量X与因变量Y
X = deliveryData[:, :-1]
Y = deliveryData[:, -1]

print("X:", "\n", X)
print("Y:", "\n", Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("coefficients:", regr.coef_)  # 系数
print("intercept:", regr.intercept_)  # 截距
print("估计多元回归方程：y=%0.3f+%0.3fx1+%0.3fx2" % (regr.intercept_, regr.coef_[0], regr.coef_[1]))

xPred = [[102, 6]]
yPred = regr.predict(xPred)
print("predicted y:", yPred)
