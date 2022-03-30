from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from DM import DataMining

d=DataMining()
lr = LogisticRegression(penalty='l2',C=100, random_state=1,max_iter=100,solver='liblinear')
lr.fit(d.X_train_std, d.y_train)
# print("Class:", lr.classes_)
# print("Coef:", lr.coef_)
# print("intercept", lr.intercept_)
# print("n_iter", lr.n_iter_)

# plot_decision_region(X_train_std, y_train, classifier=lr, resolution=0.02)
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()
d.predict(lr)
