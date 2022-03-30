import pandas as pd
from sklearn import datasets, metrics
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler


class DataMining:

    def __init__(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        # print("Class labels:", np.unique(y))
        # 切分训练数据和测试数据
        from sklearn.model_selection import train_test_split

        ## 30%测试数据，70%训练数据，stratify=y表示训练数据和测试数据具有相同的类别比例
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
        # 均值方差法进行数据归一化
        sc = StandardScaler()
        ## 估算训练数据中的mu和sigma
        sc.fit(self.X_train)
        ## 使用训练数据中的mu和sigma对数据进行标准化
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

    def plot_decision_region(self,X, y, classifier, resolution=0.02):
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx],
                        marker=markers[idx],
                        label=cl,
                        edgecolors='black')

    def predict(self,sk):
        # 预测
        # 预测训练集在各个类别的概率
        # print("训练集在各个类别的预测概率为：\n", sk.predict_proba(self.X_train_std))
        # print("\n============================")
        # 获得训练集的分类标签
        train_predict=sk.predict(self.X_train_std)
        # print("\n============================")
        # 预测测试集在各个类别的概率
        # print("测试集在各个类别的预测概率为：\n", sk.predict_proba(self.X_test_std))
        # print("\n============================")
        # 获得测试集的分类标签
        test_predict=sk.predict(self.X_test_std)
        # print("\n============================")


        # 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
        print('对训练集预测')
        # print('The ACC is:', metrics.accuracy_score(self.y_train, train_predict))
        # print('The Recall is:', metrics.recall_score(self.y_train, train_predict, average=None))
        # print('The Precision is:', metrics.precision_score(self.y_train, train_predict,average=None))
        # print('The Precision is:', metrics.precision_score(self.y_train, train_predict,average='micro'))
        print('The Precision is:', metrics.precision_score(self.y_train, train_predict,average='macro'))
        # print('The F-score is:', metrics.f1_score(self.y_train, train_predict,average='macro'))
        print('对测试集预测')
        # print('The ACC is:', metrics.accuracy_score(self.y_test, test_predict))
        # print('The Recall is:', metrics.recall_score(self.y_test, test_predict,average='weighted'))
        print('The Precision is:', metrics.precision_score(self.y_test, test_predict,average='macro'))
        # print('The F-score is:', metrics.f1_score(self.y_test, test_predict,average='binary'))


