# 决策树分类器
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
# from DataMining import X_train_std, y_train, plot_decision_region, predict
from DM import DataMining
# 决策树可视化
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz



# d.plot_decision_region(d.X_train_std, d.y_train, classifier=tree, resolution=0.02)
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# dot_data = export_graphviz(tree, filled=True, class_names=['Setosa', 'Versicolor', 'Virginica'],
#                            feature_names=['petal_length', 'petal_width'], out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('./tree.png')
d = DataMining()

tree = DecisionTreeClassifier(criterion='gini', max_depth=10,min_samples_split=0.1, random_state=1)
tree.fit(d.X_train_std, d.y_train)
d.predict(tree)

tree = DecisionTreeClassifier(criterion='gini', max_depth=10,min_samples_split=0.5, random_state=1)
tree.fit(d.X_train_std, d.y_train)
d.predict(tree)

tree = DecisionTreeClassifier(criterion='gini', max_depth=10,min_samples_split=10, random_state=1)
tree.fit(d.X_train_std, d.y_train)
d.predict(tree)