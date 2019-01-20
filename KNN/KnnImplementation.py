# Example of KNN implemented from Scratch in Python
import csv
import random
import math
import operator


# 加载数据集，按split比例分为训练集和测试集
def load_dataset(split, training_set, test_set):
    with open("iris.data.txt", "rt") as csv_file:
        lines = csv.reader(csv_file)
        # dataset = list(lines)
        # for x in range(len(dataset) - 1):
        #     for y in range(4):
        #         dataset[x][y] = float(dataset[x][y])
        #     if random.random() < split:
        #         training_set.append(dataset[x])
        #     else:
        #         test_set.append(dataset[x])
        for row in lines:
            if random.random() < split:
                training_set.append(row)
            else:
                test_set.append(row)


# 计算两个实例间的欧几里得距离
def euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow(float(instance1[i]) - float(instance2[i]), 2)
    return math.sqrt(distance)


# 得到与测试实例最邻近的K个邻居
def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for i in range(len(training_set)):
        distance = euclidean_distance(training_set[i], test_instance, length)
        distances.append((training_set[i], distance))
    distances.sort(key=lambda dis: dis[1])
    neighbors = []
    for j in range(k):
        neighbors.append(distances[j][0])
    return neighbors


# 以出现频率最高的类别作为测试数据的预测分类
def get_response(neighbors):
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


# 计算分类准确度
def get_accuracy(test_set, predictions):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return correct / float(len(test_set))


# 主方法
def main():
    training_set = []
    test_set = []
    split = 0.67
    load_dataset(split, training_set, test_set)
    print('Tran Set:' + repr(len(training_set)))
    print('Test Set:' + repr(len(test_set)))
    predictions = []
    k = 3
    for i in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[i], k)
        result = get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + result + ', actual=' + test_set[i][-1])
    accuracy = get_accuracy(test_set, predictions) * 100
    print('Accuracy:' + str('%.2f' % accuracy) + '%')


main()
