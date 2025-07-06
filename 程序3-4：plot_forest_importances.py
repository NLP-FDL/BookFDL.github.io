"""
演示内容：采用ExtraTreesClassifier进行特征的重要性排序
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier #特别随机树

# Build a classification task using 3 informative features #n_redundant : int, optional (default=2) The number of redundant features. These features are generated as random linear combinations of the informative features. n_repeated : int, optional (default=0) The number of duplicated features, drawn randomly from the informative and the redundant features.
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=4,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=200,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

          
#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)。
# [::-1]的含义是将元组或列表的内容翻转，即从大到小
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], \
    importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importance scores")
plt.bar(range(X.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
