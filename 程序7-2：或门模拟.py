# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:52:44 2021

@author: Administrator
"""

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 仅权重和偏置 与 AND不同！
    b = -0.2        # 偏置小于 权重
    tmp = np.sum(w * x) + b
    print(tmp)
    if tmp <= 0:
        return 0
    else:
        return 1


print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(0.5, 1))