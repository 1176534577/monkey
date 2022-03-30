from matplotlib import pyplot as plt
from sklearn.svm import SVC

from DM import DataMining

d = DataMining()
# svm = SVC(kernel='linear', C=1.0, random_state=1, probability=True)
# svm.fit(d.X_train_std, d.y_train)
# plot_decision_region(X_train_std, y_train, classifier=svm, resolution=0.02)
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()
# d.predict(svm)

svm = SVC(kernel='rbf', random_state=1, gamma=0.20, C=1.0, probability=True)  # 较小的gamma有较松的决策边界
svm.fit(d.X_train_std, d.y_train)
# plot_decision_region(X_train_std, y_train, classifier=svm, resolution=0.02)
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()
d.predict(svm)


svm = SVC(kernel='rbf', random_state=1, gamma=1, C=1.0, probability=True)
svm.fit(d.X_train_std, d.y_train)
d.predict(svm)


# 使用核函数对非线性分类问题建模(gamma=100)
svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0, probability=True)
svm.fit(d.X_train_std, d.y_train)
# plot_decision_region(X_train_std, y_train, classifier=svm, resolution=0.02)
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()
d.predict(svm)
