# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 09:56:26 2018

@author: ThinkPad
"""
#交叉熵
import numpy as np

predicted=np.array([12.2,  5.5,  6.9,  7.9])
label=np.array([1,    0,    0,    0])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)#axis=0表述列 axis=1表述行

print (softmax(predicted))

loss=-np.sum(label*np.log(softmax(predicted)))
print ("cross entropy loss:",loss)


