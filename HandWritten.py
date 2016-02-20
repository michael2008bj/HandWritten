# File: HandWritten.py
# Author: Lvzhe
# Data: 2016.02.20
# Desc: Test the classification algorithms I've learned using the MNIST Database of Handwritten Digits
import numpy as np
import getDistance

# Function: LoadDataSet
# Desc: Load the train data and test data, then despite into pixels and labels
# Input: None
# Output: train_data_pixels, train_data_labels, test_data_pixels, test_data_labels <numpy.ndarray>
def LoadDataSet():
	train_data = np.loadtxt('./mnist_train.csv',delimiter= ',')
	#test_data = np.loadtxt('./mnist_test.csv',delimiter= ',')
	#train_data = np.loadtxt('./sample.csv',delimiter= ',')
	test_data = np.loadtxt('./sample.csv',delimiter= ',')
	train_data_labels = train_data[:,0]
	train_data_pixels = train_data[:,1:]
	test_data_labels = test_data[:,0]
	test_data_pixels = test_data[:,1:]
	print(type(train_data))
	return train_data_pixels, train_data_labels, test_data_pixels, test_data_labels

# Function: PreProcessingData
# Input: data <numpy.ndarray>
# Output: data <numpy.ndarray>
# Desc: Do some pre-processing. For this version, treat the grey level less than 10 as 0,
#       larger or equal to 10 as 1, for I think that if the grey level is low, it may be just meaningless
def PreProcessingData(data):
	data[data<10] = 0
	data[data>=10] = 1
	return data

# Fuction: KNN_classification
# Input: sample<numpy.ndarray 1*N>:the sample to be classified,
#        dataset<numpy.ndarray M*N>:the training dataset,
#        labels<numpy.ndarray M*1>:the cluster labels, k<int>:the parameter k of kNN algorithm
# Output: voted_label<int>:the ID of sample's cluster name by kNN algorithm
# Desc: Classify the sample by kNN algorithm
def KNN_classification(sample, dataset, labels, k, distance_type):
	data_len = len(labels)
	# if the number of dataset is less than k, then we cannot use kNN algorithm
	if data_len<k:
		print('No Enough DataSet!')
		exit()
	# calculate the distance between the sample and all the dataset
	if "COSSIMILARITY"==distance_type:
		distance = np.array([getDistance.EucilideanMetric(sample,dataset[i])
                             for i in range(data_len)])
	# sort the distances, and get the indices of sorted distance
	sorted_dis_indices = distance.argsort()
	# to vote the finally label of the sample by the k nearest neighbors
	# using a dict to store the label(key) and votes(value)
	dict_vote = {}
	for i in range(k):
		vote_label_key = labels[sorted_dis_indices[i]]
		dict_vote[vote_label_key] = dict_vote.get(vote_label_key, 0) + 1
		# value = dict.get(key, default_value) equals to
		# if key in dict:
		#     value = dict[key]
		# else:
		#     value = default_value
	# find the label which gets the most votes, and set the label as the result
	max_vote_label_key = 0
	max_vote_value = 0
	for d_k, d_v in dict_vote.items():
		if d_v>max_vote_value:
			max_vote_label_key = d_k
			max_vote_value = d_v
	voted_label = max_vote_label_key
	return  voted_label


### main function ###
# set parameters
row_len = 28
col_len = 28
classify_type = 'KNN'
#ENUM {ALL, KNN, NAIVE_BAYES, SVM...}
pre_process = 'OFF'
#ENUM {ON, OFF}
para_K = 5
distance_type = 'COSSIMILARITY'
#ENUM {EUCILIDEAN, COSSIMILARITY, MADIS...}
print("Start!")
# load train data & test data
train_data, train_label, test_data, test_label = LoadDataSet()
print("Loading Data: Done!")
# pre-processing data
if 'ON'==pre_process:
	train_data = PreProcessingData(train_data)
	test_data = PreProcessingData(test_data)
train_data_len = len(train_label)
test_data_len = len(test_label)
predict_labels = np.array([KNN_classification(test_data[i], train_data, train_label, para_K, distance_type)
                 for i in range(test_data_len)])
print("Classifing: Done")
# compare predict_labels of test data with real test labels
error_list = predict_labels[predict_labels!=test_label]
error_num = len(error_list)
total_num = len(predict_labels)
error_percentage = float(error_num) / float(total_num)
print("Error Percentage is ",error_percentage)