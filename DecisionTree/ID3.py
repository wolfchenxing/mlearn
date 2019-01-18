from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
import pydot
from sklearn.externals import joblib
from sklearn.externals.six import StringIO


# Read in the csv file and put features into list of dict and list of class label
# 从csv文件中读取数据
# 数据预处理，sklearn要求数据输入的特征值（属性）features以及输出的类，必须是数值型的值，而不能是类别值
allElectronicsData = open("allElectronicsData.csv", "rt")
reader = csv.reader(allElectronicsData)
headers = next(reader)

print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)

# Vetorize features
# 字典特征提取器,将字典数据结构的原型特征名称采用0 1二值方式进行向量化
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print(vec.get_feature_names())
print("dummyX:")
print(str(dummyX))

print("labelList:" + str(labelList))

# 将标签矩阵二值化
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:")
print(str(dummyY))

# 采用ID3算法（以信息增益为标准）建模
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(dummyX, dummyY)
print("clf:" + str(clf))

# 将获得的决策树模型写入dot文件
with open("allElectronicsInformationGainOri.dot", "w") as f:
    tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

# 创建PDF文件
dot_data = StringIO()
tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("allElectronicsInformationGain.pdf")

# 保存模型
joblib.dump(clf, "allElectronicsInformationGainModel.pkl")

# 读取并测试
tr = joblib.load("allElectronicsInformationGainModel.pkl")
newRowX = [[1., 0., 0., 1., 0., 0., 1., 0., 0., 1.]]
predictedY = tr.predict(newRowX)
print("predictedY:" + str(predictedY))