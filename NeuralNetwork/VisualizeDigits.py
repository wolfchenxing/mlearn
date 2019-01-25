from sklearn.datasets import load_digits
import pylab as pl


digits = load_digits()
print(digits.data.shape)  # 1797个样本，每个样本包括8*8像素的图像和一个[0, 9]整数的标签

pl.gray()
pl.matshow(digits.images[0])
pl.show()
