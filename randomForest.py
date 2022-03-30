# @FileName  :randomForest.py
# @Time      :2022/3/30 11:48
# @Author    :RYJ
# @Software  :PyCharm
from sklearn.ensemble import RandomForestClassifier
from DM import DataMining
d=DataMining()
forest = RandomForestClassifier(criterion='gini',n_estimators=25,max_depth=10,min_samples_split=5,random_state=1,n_jobs=2)
forest.fit(d.X_train_std,d.y_train)
d.predict(forest)
forest = RandomForestClassifier(criterion='gini',n_estimators=25,max_depth=10,min_samples_split=0.5,random_state=1,n_jobs=2)
forest.fit(d.X_train_std,d.y_train)
d.predict(forest)
forest = RandomForestClassifier(criterion='gini',n_estimators=25,max_depth=10,min_samples_split=0.1,random_state=1,n_jobs=2)
forest.fit(d.X_train_std,d.y_train)
d.predict(forest)