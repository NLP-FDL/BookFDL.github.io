# -*- coding: utf-8 -*-
#演示目的：数据集拆分、K折交叉验证，并报告性能评价指标

#1.导入需要用到的模块 
from sklearn import cross_validation  
from sklearn import svm  
from sklearn import datasets  

#2.载入iris数据集
iris = datasets.load_iris()  
print ("iris.data.shape:", iris.data.shape,"; iris.target.shape:", iris.target.shape)

#3.对数据集进行切分
X_train, X_test, y_train, y_test = cross_validation.train_test_split\
(iris.data, iris.target, test_size=0.4 )
print ("X_train.shape:", X_train.shape, "y_train.shape:", y_train.shape)
print ("X_test.shape:",X_test.shape, "y_test.shape:",y_test.shape)  

#利用训练集进行预测，利用测试集进行性能评价
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)  
print ("SVM test score:",clf.score(X_test, y_test) )

# 使用iris数据集对linear kernel的SVM模型做5折交叉验证（CV=5）
print ("\n\n 5-fold CV:")
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print ("accuracies:",scores)
#The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 1.96))

 #By default, the score computed at each CV iteration is the score method of the estimator.
#It is possible to change this by using the scoring parameter:
#用各类的f1平均值做为score
scores = cross_validation.cross_val_score(clf, iris.data, iris.target,cv=5, scoring='f1_weighted')
print ("f1_weighted score:", scores ) 


#将参数cv设定为StratifiedKFold策略. 分层抽样Stratified 将数据集划分成k份，不同点在于，
#划分的k份中，每一份内各个类别数据的比例和原始数据集中各个类别的比例相同。
skf = cross_validation.StratifiedKFold(iris.target, n_folds=5)
scores=cross_validation.cross_val_score(clf, iris.data, iris.target, cv=skf)
print ("StratifiedKFold" , scores)
