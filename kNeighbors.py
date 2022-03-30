# @FileName  :kNeighbors.py
# @Time      :2022/3/30 11:49
# @Author    :RYJ
# @Software  :PyCharm
from sklearn.neighbors import KNeighborsClassifier
from DM import DataMining

d = DataMining()
knn = KNeighborsClassifier(n_neighbors=2, p=2, metric="minkowski")
knn.fit(d.X_train_std, d.y_train)
d.predict(knn)

knn = KNeighborsClassifier(n_neighbors=50, p=2, metric="minkowski")
knn.fit(d.X_train_std, d.y_train)
d.predict(knn)

knn = KNeighborsClassifier(n_neighbors=100, p=2, metric="minkowski")
knn.fit(d.X_train_std, d.y_train)
d.predict(knn)
