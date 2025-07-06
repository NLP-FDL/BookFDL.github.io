# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:45:02 2018

@author: ThinkPad
"""
#多项式分布朴素贝叶斯MultinomialNB

import numpy as np
from sklearn.naive_bayes import MultinomialNB
#Generate a 6 x 100 array of ints between 0 and 4, inclusive:
X = np.random.randint(5, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])

print (X)
print (Y)
clf = MultinomialNB()
clf.fit(X, Y)
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) 布尔参数fit_prior表示是否要考虑先验概率，如果是false,则所有的样本类别输出都有相同的类别先验概率。
#否则可以自己用第三个参数class_prior输入先验概率，
#或者不输入第三个参数class_prior让MultinomialNB自己从训练集样本来计算先验概率，此时的先验概率为P(Y=Ck)=mk/mP(Y=Ck)=mk/m。其中m为训练集样本总数量，mkmk为输出为第k类别的训练集样本数。
print(clf.predict(X[2:3]))#使用冒号获取连续的几个元素，如获取第2个到第3个元素，即第2个行
print (X[2:3])
#伯努利分布朴素贝叶斯
import numpy as np
from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()
clf.fit(X, Y)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
print(clf.predict(X[2:3]))

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 09:13:24 2018

@author: ThinkPad
"""
#简单示例数据集上进行GaussianNB
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
#GaussianNB(priors=None)
print(clf.predict([[-0.8, -1]]))



#iris数据集上进行GaussianNB
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
       % (iris.data.shape[0],(iris.target != y_pred).sum()))



