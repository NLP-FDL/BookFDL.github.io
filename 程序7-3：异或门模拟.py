# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:14:35 2021

@author: Administrator
"""

import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7  # 偏置大于权重
    tmp = np.sum(w * x) + b
    print(tmp)
    if tmp <= 0:
        return 0
    else:
        return 1


print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(0.5, 1))
"""
-0.7
0
-0.19999999999999996
0
-0.19999999999999996
0
0.050000000000000044
1
"""
print('-' * 100)


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 仅权重和偏置 与 AND不同！
    b = 0.7
    tmp = np.sum(w * x) + b
    print(tmp)
    if tmp <= 0:
        return 0
    else:
        return 1


print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(0.5, 1))

"""
0.7
1
0.19999999999999996
1
0.19999999999999996
1
-0.050000000000000044
0
"""

print('-' * 100)


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 仅权重和偏置 与 AND不同！
    b = -0.2
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
"""
-0.2
0

0.3
1

0.3
1

0.55
1
"""
print('-' * 100)


def XOR(x1, x2):
    s1 = NAND(x1, x2)  # 与 AND 相反  有0 就行
    s2 = OR(x1, x2)  # 只要有一个既返回 1   # 有 1就行
    y = AND(s1, s2)  # 必须两个都是1
    return y


print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))