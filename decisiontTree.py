# 决策树分类器
from sklearn.tree import DecisionTreeClassifier
from DM import DataMining

d = DataMining()

tree = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=0.1, random_state=1)
tree.fit(d.X_train_std, d.y_train)
d.predict(tree)

tree = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=0.5, random_state=1)
tree.fit(d.X_train_std, d.y_train)
d.predict(tree)

tree = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=10, random_state=1)
tree.fit(d.X_train_std, d.y_train)
d.predict(tree)
