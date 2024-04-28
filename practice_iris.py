import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

# irisはお試しのデータセット
iris = datasets.load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# targetは的中させたいアヤメの品種
df_iris["target"] = iris.target

# Dataframeはいろいろできる
print(df_iris)
print(df_iris.columns)
print(df_iris.index)
print(df_iris.values)
print(df_iris['petal width (cm)'])
print(df_iris.loc[1])
print(df_iris.describe())

# データを学習用とテスト用に分ける
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
# 分類問題なのでMLPClassifierで分類器を定義する
clf = MLPClassifier(hidden_layer_sizes=10, activation="relu", solver="adam", max_iter=2000)

# fitで分類器をつくる
clf.fit(data_train, target_train)

# 実際に予測させて、目標と比較する
print(clf.score(data_train, target_train))
print(clf.predict(data_test))
print(target_test)

# matplotlib.pyplotで学習過程を可視化する
plt.plot(clf.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()