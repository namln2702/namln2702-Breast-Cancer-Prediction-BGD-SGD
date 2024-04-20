import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from BGD import BGD
from SGD import SGD

df = pd.read_csv('data.csv')
df['diagnosis']=df['diagnosis'].map({'B':0,'M':1})
df.drop('id',axis=1, inplace=True)
df.drop('Unnamed: 32',axis=1, inplace=True)



X_train, X_test, y_train, y_test = train_test_split(
                df.drop('diagnosis', axis=1),
                df['diagnosis'],
                test_size=0.2,
                random_state=42)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

y_train = y_train.values.reshape(1,-1)

lrs = [0.0001, 0.0003, 0.001]

for lr in lrs :

    clf = SGD(lr)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)


    def accuracy(y_test, y_pred):
        return np.sum(y_test==y_pred)/ len(y_test)

    acc = accuracy(y_test, y_pred)

    print("Accuracy SGD:", acc , "     lr = ", clf.lr)


for lr in lrs :

    clf = BGD(lr)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)


    def accuracy(y_test, y_pred):
        return np.sum(y_test==y_pred)/ len(y_test)

    acc = accuracy(y_test, y_pred)

    print("Accuracy Batch gradient descent:", acc , "     lr = ", clf.lr)



