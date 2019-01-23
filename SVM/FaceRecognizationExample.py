# from __future__ import print_function
# __future__模块，把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性
# 本地python版本为3.7.2

from time import time
import logging
import matplotlib.pylab as plt

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


print(__doc__)

# Display progress logs on stdout
# 打印程序进展日志，格式为“时间 信息”
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
# route:~/scikit_learn_data/lfw_home.
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel positions info is ignored by this model)
X = lfw_people.data  # 特征向量矩阵，每一行是个实例，每一列是个特征值
n_features = X.shape[1]  # 维度，即人脸特征值的个数，shape[1]列数

# the label to predict is the id of the person
# 提取每个实例对应每个人脸，目标分类标记，不同的人的身份
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]  # 多少类，shape[0]行数

print("Total dataset size:")
print("n_samples: %d" % n_samples)  # 实例的个数
print("n_features: %d" % n_features)  # 特征向量的维度
print("n_classes: %d" % n_classes)  # 总共有多少人


# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
# 拆分数据，分成训练集和测试集，其中测试集占25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
# PCA数据降维，提取150个特征值
n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)  # 训练PCA模型
print("Train PCA in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))  # 提取出来特征值之后的矩阵

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)  # 将训练集与测试集降维
X_test_pca = pca.transform(X_test)
print("Done PCA in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model
print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],  # C是对错误的惩罚，惩罚参数C(越大,边界越硬)
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }  # gamma核函数里多少个特征点会被使用}
# 对参数尝试不同的值，共5*6=30种组合
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
print("Done fitting in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("Done predicting in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))  # 生成分类指标的文本报告
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))  # 生成矩阵，有多少个被分为此类


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue: %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()
