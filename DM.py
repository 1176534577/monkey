from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataMining:

    def __init__(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        # 切分训练数据和测试数据
        ## 30%测试数据，70%训练数据，stratify=y表示训练数据和测试数据具有相同的类别比例
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=1,
                                                                                stratify=y)
        # 均值方差法进行数据归一化
        sc = StandardScaler()
        ## 估算训练数据中的mu和sigma
        sc.fit(self.X_train)
        ## 使用训练数据中的mu和sigma对数据进行标准化
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

    def predict(self, sk):
        # 预测
        train_predict = sk.predict(self.X_train_std)
        test_predict = sk.predict(self.X_test_std)
        print('对训练集预测')
        print('The Precision is:', metrics.precision_score(self.y_train, train_predict, average='macro'))
        print('对测试集预测')
        print('The Precision is:', metrics.precision_score(self.y_test, test_predict, average='macro'))
