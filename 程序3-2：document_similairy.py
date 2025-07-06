# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 20:37:05 2017

@author: ThinkPad
"""

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
'Jobs was the chairman of Apple Inc., and he was very famous',
'I like to use apple computer',
'And I also like to eat apple'

] 

vectorizer =CountVectorizer()
counts = vectorizer.fit_transform(corpus).todense()
for x,y in [[0,1],[0,2],[1,2]]:
	dist = euclidean_distances(counts[x],counts[y])
	print('document{} vs. document{} distance is {}'.format(x,y,dist))
print 
for x,y in [[0,1],[0,2],[1,2]]: 
	sim = cosine_similarity(counts[x],counts[y])
	print('document{} vs. document{} similarity is {}'.format(x,y,sim))