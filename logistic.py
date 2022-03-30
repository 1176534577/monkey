from sklearn.linear_model import LogisticRegression
from DM import DataMining

d = DataMining()

lr = LogisticRegression(penalty='l2', C=0.1, random_state=1, max_iter=100, solver='liblinear')
lr.fit(d.X_train_std, d.y_train)
d.predict(lr)

lr = LogisticRegression(penalty='l2', C=1, random_state=1, max_iter=100, solver='liblinear')
lr.fit(d.X_train_std, d.y_train)
d.predict(lr)

lr = LogisticRegression(penalty='l2', C=100, random_state=1, max_iter=100, solver='liblinear')
lr.fit(d.X_train_std, d.y_train)
d.predict(lr)
