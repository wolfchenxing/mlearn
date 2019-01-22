import numpy as np
import pylab
from sklearn import svm

# we create 40 separable points
np.random.seed(0)  # 每次生成的随机数相同
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]  # np.r_按列连接两个矩阵，np.c_按行连接两个矩阵
Y = [0] * 20 + [1] * 20

# fit the model
clf = svm.SVC(kernel="linear")
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]  # 权值向量
a = -w[0]/w[1]
xx = np.linspace(-5, 5)  # 创建-5到5的等差数列，默认为50个样本数
yy = a*xx - (clf.intercept_[0])/w[1]  # clf.intercept_[0]为偏置b

# plot the parallels to the separating hyperplane that pass through the support vectors
b_down = clf.support_vectors_[0]
yy_down = a*xx + (b_down[1] - a*b_down[0])
b_up = clf.support_vectors_[-1]
yy_up = a*xx + (b_up[1] - a*b_up[0])

'''
    In scikit-learn coef_ attribute holds the vectors of the separating hyperplanes for linear models. 
It has shape (n_classes, n_features) if n_classes > 1 (multi-class one-vs-all) and (1, n_features) 
for binary classification.
    In this toy binary classification example, n_features == 2, hence w = coef_[0] is the vector orthogonal 
to the hyperplane (the hyperplane is fully defined by it + the intercept).
    To plot this hyperplane in the 2D case (any hyperplane of a 2D plane is a 1D line), 
we want to find a f as in y = f(x) = a.x + b. In this case a is the slope of the line and can be computed by 
a = -w[0] / w[1].
'''

# plot the line, the points, and the nearest vectors to the plane
pylab.plot(xx, yy, 'k-')
pylab.plot(xx, yy_down, 'k--')
pylab.plot(xx, yy_up, 'k--')

pylab.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80)
pylab.scatter(X[:, 0], X[:, 1], c=Y, cmap=pylab.cm.Paired)

pylab.axis("tight")
pylab.show()
