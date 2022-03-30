from sklearn.svm import SVC
from DM import DataMining

d = DataMining()

svm = SVC(kernel='rbf', random_state=1, gamma=0.20, C=1.0, probability=True)  # 较小的gamma有较松的决策边界
svm.fit(d.X_train_std, d.y_train)
d.predict(svm)

svm = SVC(kernel='rbf', random_state=1, gamma=1, C=1.0, probability=True)
svm.fit(d.X_train_std, d.y_train)
d.predict(svm)

# 使用核函数对非线性分类问题建模(gamma=100)
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0, probability=True)
svm.fit(d.X_train_std, d.y_train)
d.predict(svm)
