from __future__ import absolute_import, division, print_function #needed for methon learnNerves
import numpy as np
from six.moves import cPickle as pickle
import os
import glob
from datetime import datetime
import math
import time
import cv2
#from configFile import Config
#learn nerves
from random import randint
import string
import random
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from helpers import *
from matplotlib import colors
import pylab as pl
'''
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
from keras.regularizers import l2, activity_l2
from six.moves import xrange  # pylint: disable=redefined-builtin
import operator

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

#import tflearn.datasets.oxflower17 as oxflower17
import tensorflow as tf
'''
# imu x kordinate pridedu ir atimu po puse plocio...
# iskerpu si kvadrata pagal pixelius is nuotraukos ir is kaukes ir issaugau i atitinkamas papkes

class Nerves():
	def __init__(self, Config):
		self.config = Config
		#self = Config
		#print(self,'selfff')
		
	def getElipseData(self):		
		with open(self.config.pickle_file, 'rb') as f:
			save = pickle.load(f)
			self.elipseList = save['elipseList']
			self.filesList = save['filesList']
			del save

	def extractNerves(self):
		self.getElipseData()
		i= 0
		for xy,centers,something in self.elipseList:
			cropXFrom = xy[0] - (self.config.maxWidth/2)
			cropXTo = xy[0] + (self.config.maxWidth/2)
			cropYFrom = xy[1] - (self.config.maxHeight/2)
			cropYTo = xy[1] + (self.config.maxHeight/2)
			
			imageMaskbase = self.extractFileName(self.filesList[i])
			
			mask = cv2.imread(self.filesList[i], -1)
			crop_mask = mask[cropYFrom:cropYTo, cropXFrom:cropXTo]
			cv2.imwrite(self.config.maskDir + imageMaskbase, crop_mask)
			
			imageName = self.getImageNameFromMaskPath(self.filesList[i])
			image = cv2.imread(self.config.trainDir + imageName)
			crop_img = image[cropYFrom:cropYTo, cropXFrom:cropXTo]
			cv2.imwrite(self.config.imgDir + imageName, crop_img)
			i = i+1
		
	def extractNotNerves(self):
		files = self.getAllMaskFiles()
		i= 0
		for oneFile in files:
			image = cv2.imread(oneFile, -1)
			empty = self.checkIfMasIsEmpty(image)
			if(empty == True):
				cropXFrom = randint(0,self.config.fullImageWidth-self.maxWidth)
				cropYFrom = randint(0,self.config.fullImageHeight-self.maxHeight)
				cropXTo = cropXFrom + self.config.maxWidth
				cropYTo = cropYFrom + self.config.maxHeight
				'''
				cropXFrom = self.defaultCentroidX - (self.config.maxWidth/2)
				cropXTo = self.defaultCentroidX + (self.config.maxWidth/2)
				cropYFrom = self.defaultCentroidY - (self.config.maxHeight/2)
				cropYTo = self.defaultCentroidY + (self.config.maxHeight/2)
				'''
				imageName = self.getImageNameFromMaskPath(oneFile)
				image = cv2.imread(self.config.trainDir + imageName)
				crop_img = image[cropYFrom:cropYTo, cropXFrom:cropXTo]
				cv2.imwrite(self.config.imgFalseDir + imageName, crop_img)
				if(i > 2320):
					return True
				i += 1
				
	def loadTrainingData(self):
		with open(self.config.trainNervesPickle, 'rb') as f:
			save = pickle.load(f)
			self.train_dataset = save['dataset']#[1:self.max_steps]
			self.train_dataset2 = save['dataset2']
			self.train_labels = save['labels']#[1:self.max_steps]
			#print ('Full training set', save['train_dataset'].shape)
			del save
		plt.figure(1, figsize=(20,20))
		fromI = 70
		toI = 105
		found = 0
		print(self.train_labels.shape,'self.train_labels')
		for index, itemList in enumerate(self.train_labels[:,:,1]):
			for index2, item in enumerate(itemList):
				if(found ==0 and item == 1):
					found = 1
					print(index,'inde55x',index2,'ind33ex2')
		'''
		for index, item in enumerate(self.train_labels[:,0,0]):
			if(found ==0 and item == 0):
				found = 1
				print(index,'index')
		#print(self.train_dataset[0,:])
		
		print(self.train_dataset.shape,'self.train_dataset')
		image = np.reshape(self.train_dataset[0,:,1], (5,7))#[i,0:35] 7,5
		plt.hexbin(image)
		plt.show()
		
		pl.pcolor(hist2D)
		pl.colorbar()
		pl.xlim([0,hist2D.shape[1]])
		pl.ylim([0,hist2D.shape[0]])
		'''
		
		fullImages = self.getAllTrainImgFiles()
		stepForWindow = 32
		#print(fullImages,'fullImages')
		xList = [0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192, 0, 32, 64, 96, 128, 160, 192]
		yList = [0, 0, 0, 0, 0, 0, 0, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128]

		for i in xrange(fromI,toI,6):
			batch_labels = np.zeros(shape=(35,1),dtype=np.float32)	
			blackImage = np.zeros(shape=(210,290),dtype=np.float32)	
			for index, item in enumerate(self.train_labels[i,:,1]):
				if(item == 1):
					blackImage[xList[index]:xList[index]+70, yList[index]:yList[index]+60] = 255
					'''
					self.train_labels[i,index-1,0] = 0
					self.train_labels[i,index-1,1] = 1	
					self.train_labels[i,index,0] = 1	
					self.train_labels[i,index,1] = 0
					'''
					continue
			

			image = cv2.imread(fullImages[i], 0)
			image = cv2.resize(image,(self.config.fullImageWidth//2, self.config.fullImageHeight//2), interpolation = cv2.INTER_CUBIC)
			maskImagePath = self.getMaskNameFromImagePath(fullImages[i])
			imageMask = cv2.imread(self.config.trainDir + maskImagePath, 0)
			imageMask = cv2.resize(imageMask,(self.config.fullImageWidth//2, self.config.fullImageHeight//2), interpolation = cv2.INTER_CUBIC)
			'''
			imageNumber = 1
			targetSum = 0
			target = 0
			for (xx, yy, window) in sliding_window(image, stepForWindow, ( self.config.maxWidth,  self.config.maxHeight)):
				# if the window does not meet our desired window size, ignore it
				if window.shape[0] !=  self.config.maxHeight or window.shape[1] !=  self.config.maxWidth:
					continue
				#findout label for this part of image
				windowMask = imageMask[xx:xx+ self.config.maxWidth,yy:yy+ self.config.maxHeight]
				#print(np.sum(windowMask[:,:]),'pirma suma', (configModel.maxWidth*configModel.maxHeight)/2.5 ,'turi buti didesne uz sia')
				if(np.sum(windowMask[:,:]) > targetSum):
					target = imageNumber
					targetSum = np.sum(windowMask[:,:])
				imageNumber += 1
			batch_labels = batch_labels[:imageNumber-1,::]							
			if(target > 0):
				batch_labels[target,0] = 1
			'''
			image2 = np.reshape(self.train_dataset[i,:,0], (5,7))
			print(image2,'image2')
			image3 = np.reshape(self.train_dataset[i,:,1], (5,7))
			image4 = np.reshape(self.train_dataset2[i,:]*255, (5,7))
			'''
			self.train_labels[i,0,1] = 1
			self.train_labels[i,34,1] = 1
			batch_labels[0] = 1
			batch_labels[34] = 1
			'''
			image5 = np.reshape(self.train_labels[i,:,1]*255, (5,7))
			print(imageMask.shape,'imageMask')
			
			#image6 = np.reshape(imageMask, (5*7))
			#image6 = np.reshape(image6, (5,7))
			#image6 = np.reshape(batch_labels*255, (5,7))
			#print(image2,'image2',image3,'image3')
			plt.subplot(6,6,i+1-fromI)
			plt.title('image ' + str(fullImages[i]))
			plt.imshow(image, interpolation="nearest", cmap="gray")
			plt.subplot(6,6,i+2-fromI)
			plt.imshow(image2, interpolation="nearest", cmap="gray")
			plt.subplot(6,6,i+3-fromI)
			plt.imshow(image3, interpolation="nearest", cmap="gray")
			plt.subplot(6,6,i+4-fromI)
			plt.imshow(image4, interpolation="nearest", cmap="gray")
			plt.subplot(6,6,i+5-fromI)
			plt.imshow(blackImage, interpolation="nearest", cmap="gray")		
			plt.subplot(6,6,i+6-fromI)
			plt.title('imageMask mask' )
			plt.imshow(image5, interpolation="nearest", cmap="gray")		
			'''
			plt.subplot(4,4,i+1-fromI)
			plt.title('image ' + str(i))
			analysedImage = np.zeros(shape=(7*5),dtype=np.float32)
			analysedImage2 = np.zeros(shape=(7*5),dtype=np.float32)
			analysedImage2 = self.train_labels[i,:,1]
			#print(analysedImage,'analysedImage')
			analysedImage[0:9] = self.train_dataset[i,0:9]
			analysedImage[9:13] = self.train_dataset[i,14:18]
			analysedImage[13:18] = self.train_dataset[i,23:28]
			analysedImage[18:27] = self.train_dataset[i,32:41]
			analysedImage[27:34] = self.train_dataset[i,42:49]
			analysedImage[34] = self.train_dataset[i,55]
			'''
			'''
			print(self.train_dataset[i,0:8],'self.train_dataset[i,0:8]',
			self.train_dataset[i,14:18],'self.train_dataset[i,14:18]',
			self.train_dataset[i,23:28],'self.train_dataset[i,14:17]',
			self.train_dataset[i,32:41],'self.train_dataset[i,32:41]',
			self.train_dataset[i,42:49],'self.train_dataset[i,23:27]')
			'''
			'''
			#image = np.reshape(self.train_dataset[i], (9,14))#[i,0:35] 7,5
			image = np.reshape(analysedImage, (5,7))#[i,0:35] 7,5
			imageLabels = np.reshape(analysedImage2, (5,7))
			plt.imshow(image, interpolation="nearest", cmap="gray")
			plt.subplot(4,4,i-fromI)
			plt.imshow(imageLabels, interpolation="nearest", cmap="gray")
			'''
		
		plt.show()
		
	def loadTrainingDataFullImages(self, pickelFile):
		with open(pickelFile, 'rb') as f:
			save = pickle.load(f)
			#print(save,'saveeee')
			self.train_dataset = save['train_dataset']#[1:self.max_steps]
			self.train_labels = save['train_labels']#[1:self.max_steps]
			del save
		#reshape images
		self.train_dataset /=255
		print(self.train_dataset.shape,'train_dataset.shape')
		print(self.train_labels.shape,'train_labels.shape')
		
	def prepareDataTolearn(self):
		self.loadTrainingData()
		
	def separateValidationAndTestSet(self):
		kf = KFold(self.train_labels.shape[0]-1, n_folds=3, shuffle=True, random_state=self.config.randomState)
		for train, test in kf:
			validIndex = test[0:len(test)/2]
			testIndex = test[(len(test)/2)+1::]

		self.test_dataset = self.train_dataset[testIndex,::]
		self.test_labels = self.train_labels[testIndex,:]
		self.valid_dataset = self.train_dataset[validIndex,::]
		self.valid_labels = self.train_labels[validIndex]
		self.train_dataset = self.train_dataset[train,::]
		self.train_labels =  self.train_labels[train]
		print(self.train_dataset.shape,'train_dataset.shape')
		print(self.train_labels.shape,'train_labels.shape')
		print(self.valid_dataset.shape,'valid_dataset.shape')
		print(self.valid_labels.shape,'valid_labels.shape')
		print(self.test_dataset.shape,'test_dataset.shape')
		print(self.test_labels.shape,'test_labels.shape')
		'''
		shuffledData = np.c_[self.train_dataset , self.train_labels]
		np.random.shuffle(shuffledData)
		dataset = shuffledData[:,0 : self.fullImageHeight * self.fullImageWidth]
		labels = shuffledData[:,self.fullImageHeight * self.fullImageWidth ::]
		datasetLenght = dataset.shape[0]		
		validIndexTo = datasetLenght * self.partForValidation
		testIndexTo = datasetLenght * self.partForTest
		self.test_dataset = dataset[0:testIndexTo,::]
		self.test_labels = labels[0:testIndexTo]
		self.valid_dataset = dataset[testIndexTo+ 1: testIndexTo + validIndexTo,::]
		self.valid_labels = labels[testIndexTo+ 1: testIndexTo + validIndexTo]
		self.train_dataset = dataset[testIndexTo + validIndexTo + 1 ::,::]
		self.train_labels =  labels[testIndexTo + validIndexTo + 1 ::]
		'''

		
	def convertToOneHot(self, labels, numClasses):
		returnOneHot = np.zeros((len(labels), numClasses), dtype=np.int32)
		i = 0
		for label in labels:
			if(label == 1):
				returnOneHot[i,0] = 1
			else:
				returnOneHot[i,1] = 1
			i += 1
		return returnOneHot
		
	def trainingDataToPickle(self, width, height, mode = 'nerves', fileFormat = 'tif'):

		if os.path.exists(self.config.trainNervesPickle):
			# You may override by setting force=True.
			print('%s already present - Skipping pickling.' % self.config.trainNervesPickle)
			return []
		print('Pickling %s.' % self.config.trainNervesPickle)
		
		if(mode == 'nerves'):
			files = glob.glob(self.config.imgDir + '*.' + fileFormat)
			filesNegative = glob.glob(self.config.imgFalseDir + '*.' + fileFormat)
			dataset = np.ndarray(shape=(len(files) + len(filesNegative), height * width),
								dtype=np.float32)
			labels = np.ndarray(shape=(len(files) + len(filesNegative)),dtype=np.int32)
			'''
			Sugadinti failai 
			./input/nerves/33_15.tif image_file
			./input/nerves/33_5.tif image_file
			./input/nerves/33_42.tif image_file
			./input/nerves/33_70.tif image_file
			./input/nerves/33_6.tif image_file
			'''
		else:
			files = glob.glob(self.config.trainDir + '*[0-9].'+ fileFormat)
			dataset = np.ndarray(shape=(len(files), height * width),
								dtype=np.float32)
			labels = np.ndarray(shape=(len(files)),dtype=np.int32)
		num_images = 0
		
		#read data
		for image_file in files:
			try:
				image = cv2.imread(image_file, 0)
				#print(mode,'mode',image.shape[0],image.shape[1],'image.shape',height, width,'height * width')
				if(image is not None):
					if(mode == 'nerves'):
						image = cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
						dataset[num_images, :] = image.reshape(width* height)
						labels[num_images] = 1
					else:
						flbase = os.path.basename(image_file)
						mask_path = "./input/trainChanged/" + flbase[:-4] + "_mask." + fileFormat
						mask = cv2.imread(mask_path, 0)
						if(mask is not None):
							image = cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
							dataset[num_images, :] = image.reshape(height * width)
							emptyMask = self.checkIfMasIsEmpty(mask)
							labels[num_images] = int(emptyMask)
						else:
							num_images = num_images - 1	
							print('Could not read mask:', mask_path, '- it\'s ok, skipping.')
					num_images = num_images + 1
				else:
					print('Could not read: ' + image_file,'image_file')
			except IOError as e:
				print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
		if(mode == 'nerves'):
			for image_file in filesNegative:
				try:
					image = cv2.imread(image_file, 0)
					if(image is not None):
						image = cv2.resize(image,(width, height), interpolation = cv2.INTER_CUBIC)
						dataset[num_images, :] = image.reshape(height * width)
						labels[num_images] = 0
						num_images = num_images + 1
					else:
						print(image_file,'image_file2222')
				except IOError as e:
					print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
			# end read data
				
		dataset = dataset[0:num_images, :]
		labels = labels[0:num_images]
		print(dataset.shape,'new shape')
		print(labels.shape,'labels.shape')
		
		if(mode == 'nerves'):
			#shuffle and separate data
			shuffledData = np.c_[dataset , labels]
			np.random.shuffle(shuffledData)
			dataset = shuffledData[:,0 : height * width]
			labels = shuffledData[:,height * width ::]
			datasetLenght = dataset.shape[0]		
			validIndexTo = datasetLenght * self.config.partForValidation
			testIndexTo = datasetLenght * self.config.partForTest
		try:
			'''

							'train_dataset': dataset,
				'train_labels' : labels,
			'''
			f = open(self.config.trainNervesPickle, 'wb')
			save = {
				'valid_dataset': dataset[0:int(validIndexTo),:],
				'valid_labels' : labels[0:int(validIndexTo)],
				'test_dataset': dataset[int(validIndexTo)+1:int(validIndexTo) + int(testIndexTo),:],
				'test_labels' : labels[int(validIndexTo)+1:int(validIndexTo) + int(testIndexTo)],
				'train_dataset': dataset[int(validIndexTo) + int(testIndexTo)+1::,:],
				'train_labels' : labels[int(validIndexTo) + int(testIndexTo)+1::],
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print('Unable to save data to elis:', e)
			raise

		
	def extractFileName(self, path):
		imagebase = os.path.basename(path)
		return imagebase

	def getImageNameFromMaskPath(self, path):
		base = self.extractFileName(path)
		imageName = base[:-9] + '.tif'
		return imageName

	def getMaskNameFromImagePath(self, path):
		base = self.extractFileName(path)
		imageName = base[:-4] + '_mask.tif'
		return imageName
				
	def getAllMaskFiles(self):
		return glob.glob(self.config.trainDir + "*_mask.tif")
		
	def getAllTrainImgFiles(self):
		return glob.glob(self.config.trainDir + "*[0-9].tif")
		
	def checkIfMasIsEmpty(self, mask):
		return np.sum(mask[:,:]) == 0
