"""
使用主成分分析法（principal component analysis） 进行特征降维及数据的可视化
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
data = load_iris()

X = data.data
y = data.target

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X)

red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_X)):
	if y[i] == 0:
	   red_x.append(reduced_X[i][0])
	   red_y.append(reduced_X[i][1])
	elif y[i] == 1:
	   blue_x.append(reduced_X[i][0])
	   blue_y.append(reduced_X[i][1])
	else:
	   green_x.append(reduced_X[i][0])
	   green_y.append(reduced_X[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')# x号
plt.scatter(blue_x, blue_y, c='b', marker='D')#diamond 钻石
plt.scatter(green_x, green_y, c='g', marker='.')#显示为点
plt.show()