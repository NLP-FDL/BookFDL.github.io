# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:32:52 2021
https://wiki.swarma.org/index.php?title=%E6%84%BF%E6%9C%9B%E6%A0%91&oldid=9229
@author: lenovo
基于教材的实例
"""
import numpy as np
training_set = [[3,3],[4,4],[0,1]]
Y = [1,1,-1]
w = [0,0]
b = 0
n = 1
#check给出w，b，找到一个此时的误分点，如果没有，说明算法收敛
def check(w,b):
    C = []
    for e in training_set:
        x1 = np.array(e).reshape((-1,1))
        w1 = np.array(w)
        c = np.dot(w1,x1)
        y = Y[training_set.index(e)]
        if y*(c+b)<=0:
            C.append(e[0])
            C.append(e[1])
            break
        else:
            continue
    return C
def update(C,w,b):
    D = []
    y = Y[training_set.index(C)]
    x = np.array(C)
    D.append(w + n*y*x)
    D.append(b + n*y)
    return D

while True:
    C = check(w, b)
    if len(C)==0:
        break
    else:
        w = update(C,w,b)[0]
        b = update(C,w,b)[1]
print('算法已经收敛')
print('y = sign(%d*x1 + %d*x2 + %d)'%(w[0],w[1],b))