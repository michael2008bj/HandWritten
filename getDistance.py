# File: getDistance.py
# Auther: Lvzhe
# Date: 2016.02.20
# Desc: some algorithms of calculating distance
import numpy as np
import pandas as pd
import math
from numpy.linalg import inv
# Fuction: cosdistance
# Input: vec1<numpy.ndarray N*1 or 1*N>, vec2<numpy.ndarray same size as vec1>
# Output: the cosine similarity of the 2 vectors<float>
# Desc: calculate the cosine similarity of the 2 vectors
def cosdistance(vec1, vec2):
	a = np.dot(vec1,vec2)
	b = math.sqrt(vec1.dot(vec1))
	c = math.sqrt(vec2.dot(vec2))
	return a / (b*c)

# Fuction: MaDistance
# Input: vec1<numpy.ndarray 1*N>, matrix_sample<numpy.ndarray M*N>
# Output: dis<float>:the Mahalanobis distance
# Desc: calculate the Mahalanobis distance
def MaDistance(vec1, matrix_sample):
	segama = np.cov(matrix_sample)
	inv_mat = inv(segama)
	dis_list = [math.sqrt((vec1 - matrix_sample[c]).dot(inv_mat).dot((vec1 - matrix_sample[c]).T))
	            for c in matrix_sample]
	dis = np.array(dis_list)
	return dis

def MASimDis(vec, matrix, MA):
	matrix_arr = np.array(matrix)
	[row, column] = matrix_arr.shape
	matrix_ma = np.zeros((row-MA+1,column))
	v_index = []
	print(vec.shape)
	for i in np.arange(row-MA+1):
		matrix_ma[i,:] = np.mean(matrix_arr[i:i+MA,:],axis=0)
		v_index.append(pd.Series.mean(vec[i:i+MA]))
	dis = []
	v_s = pd.Series(v_index)
	print(type(v_s))
	for i in np.arange(row-MA+1):
		v = matrix_ma[:,i]
		v_vs = pd.Series(v)
		dis.append(Similarity(v_s, v_vs))
	return dis

def Similarity(vec1, vec2):
	return vec1.corr(vec2)

# Fuction: EucilideanMetric
# Input: vec1<numpy.ndarray 1*N or N*1>, vec2<numpy.ndarray same size as vec1>
# Output: Eucilidean Metric of the 2 vectors<float>
# Desc: calculate the Eucilidean metric(or say distance)
def EucilideanMetric(vec1, vec2):
	return math.sqrt(sum((vec1 - vec2)**2))
