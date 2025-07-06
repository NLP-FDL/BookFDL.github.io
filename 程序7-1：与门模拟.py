# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:42:14 2021

@author: Administrator
"""
def AND(x1, x2):
    """
    与门

    在函数内初始化参数 w1 、 w2 、 theta ，当输入的加权总和超过阈值时返回 1 ， 否则返回 0 。

    """
    # w：权重，theta：阈值，x：参数
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    #print(tmp)
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(0.5, 1))


#或者使用权重和偏置，可以像下面这样实现与门。b称为偏置，w 1 和w 2 称为权重。
#感知机会计算输入 信号和权重的乘积，然后加上偏置，如果这个值大于0则输出1，否则输出0。

import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7    # 偏置大于权重
    tmp = np.sum(w * x) + b
    #print(tmp)
    if tmp <= 0:
        return 0
    else:
        return 1


print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(0.5, 1))
