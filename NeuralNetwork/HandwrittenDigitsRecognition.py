import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from NeuralNetwork import NeuralNetwork


digits = load_digits()
X = digits.data
y = digits.target
X = (X - X.min()) / X.max()  # normalize the values to bring them into the range 0-1

X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)

# start fitting
nn = NeuralNetwork([64, 100, 10], activation='logistic')
nn.fit(X_train, labels_train, epochs=10000)

predictions = []
for i in range(X_test.shape[0]):
    o = nn.predict(X_test[i])
    predictions.append(np.argmax(o))  # argmax返回最大数的索引
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
