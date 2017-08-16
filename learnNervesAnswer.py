import cv2
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import glob
import random
from configFile import Config
from NervesFile import Nerves
from helpers import *
import tensorflow as tf

configModel = Config()
configModel.trainNervesPickle = configModel.masksPickle
configModel.batch_size = configModel.imageHeight *  configModel.imageWidth
##########################
restore = 2
##########################
if(restore == 2):
	configModel.trainDir = './input/test/'
dataModel = Nerves(configModel)

training_iters = 5630*35
display_step = 10
validationSetSize = ((training_iters//display_step)//configModel.batch_size)+1
koef = 0.95
trainIters = 5508
if(restore == 1):
	fullImages = dataModel.getAllTrainImgFiles()
	imageCount = len(fullImages)
	print(imageCount,'imageCount')
else:
	dataModel.prepareDataTolearn(True, True)
	#dataModel.train_dataset = np.concatenate((dataModel.train_dataset,dataModel.test_dataset))
	#dataModel.train_labels = np.concatenate((dataModel.train_labels,dataModel.test_labels))
	'''
	dataModel.valid_dataset = dataModel.valid_dataset[0:validationSetSize,::]
	dataModel.valid_labels = dataModel.valid_labels[0:validationSetSize]
	dataModel.train_dataset = np.concatenate((dataModel.train_dataset,dataModel.valid_dataset[validationSetSize + 1::,::]))
	dataModel.train_labels = np.concatenate((dataModel.train_labels,dataModel.valid_labels[validationSetSize +1 ::]))
	'''
	if(dataModel.train_labels.shape[0]<training_iters):
		training_iters = dataModel.train_labels.shape[0]
#print(dataModel.train_labels.shape,'shapeee',dataModel.train_dataset.shape,type(dataModel.train_labels),type(dataModel.train_dataset))

def fillFeedDict(dataset, labels, step, test= False):
	images = np.zeros(shape=(labels.shape[1],dataset.shape[2]),dtype=np.float32)
	label_batch = np.zeros(shape=(labels.shape[1],labels.shape[2]),dtype=np.float32)
	images = dataset[step,:,:]
	label_batch = labels[step,:,:]
	return images, label_batch

step = 1
# Keep training until reach max iterations
predictions = []
print(dataModel.train_dataset.shape,'dataModel.train_dataset shape')
while step  < trainIters:
	batch_x, batch_y = fillFeedDict(dataModel.train_dataset, dataModel.train_labels, step)
	temp = -1
	indexTemp = 0
	for index, item in enumerate(batch_x[:,0]):
		if(item > 0 and item > temp):
			temp = item
			indexTemp = index
	if(temp > koef):
		predictions.append(indexTemp)
	else:
		predictions.append(-1)
	step += 1
dataModel.create_submission(predictions)

