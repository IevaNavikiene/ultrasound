from __future__ import absolute_import, division, print_function #needed for methon learnNerves
import numpy as np
from six.moves import cPickle as pickle
import os
import glob
from datetime import datetime
import math
import datetime
import time
import cv2
#from configFile import Config
#learn nerves
from random import randint
import string
import random
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
# Random Shifts
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
#from random import shuffle
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
			xRandom = np.random.uniform(0, 15)
			yRandom = np.random.uniform(0, 15)
			cropXFrom = xy[0] + xRandom - (self.config.maxWidth/2)
			cropXTo = xy[0] + xRandom + (self.config.maxWidth/2)
			cropYFrom = xy[1] - yRandom - (self.config.maxHeight/2)
			cropYTo = xy[1] - yRandom + (self.config.maxHeight/2)
			
			imageMaskbase = self.extractFileName(self.filesList[i])
			'''
			mask = cv2.imread(self.filesList[i], -1)
			crop_mask = mask[cropYFrom:cropYTo, cropXFrom:cropXTo]
			cv2.imwrite(self.config.maskDir + imageMaskbase, crop_mask)
			'''
			imageName = self.getImageNameFromMaskPath(self.filesList[i])
			image = cv2.imread(self.config.trainDir + imageName)
			crop_img = image[cropYFrom:cropYTo, cropXFrom:cropXTo]
			cv2.imwrite(self.config.imgDir + 'd_' + imageName, crop_img)
			i = i+1
		
	def extractNotNerves(self):
		files = self.getAllMaskFiles()
		i= 0
		for oneFile in files:
			image = cv2.imread(oneFile, -1)
			empty = self.checkIfMasIsEmpty(image)
			if(empty == True):
				cropXFrom = randint(0,self.config.fullImageWidth-self.config.maxWidth)
				cropYFrom = randint(0,self.config.fullImageHeight-self.config.maxHeight)
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
				cv2.imwrite(self.config.imgFalseDir + 'd'+imageName, crop_img)
				#if(i > 109):#2320
				#	return True
				i += 1
				
	def loadTrainingData(self, longShortTermMemory=False, submissionFile = False):
		with open(self.config.trainNervesPickle, 'rb') as f:
			save = pickle.load(f)
			if(longShortTermMemory == False):
				self.valid_dataset = save['valid_dataset']
				self.valid_labels = save['valid_labels']
				self.test_dataset = save['test_dataset']
				self.test_labels = save['test_labels']
				#if(save['train_dataset'].shape[0] < self.config.max_steps):
				#raise Exception('Wrong parameter training_iters of class BaseTensorFlow. Max can be %d', save['train_dataset'].shape[0])
				self.train_dataset = save['train_dataset']#[1:self.max_steps]
				self.train_labels = save['train_labels']#[1:self.max_steps]
				#print ('Full training set', save['train_dataset'].shape)
			else:
				fullImages = self.getAllTrainImgFiles()
				self.photosId = []
				for number in range(0, len(fullImages)):
					flbase = os.path.basename(fullImages[number])
					self.photosId.append(flbase[:-4])
				#print(save['dataset'].shape,save['dataset2'].shape)
				self.train_dataset = save['dataset']
				#predictions = save['dataset2']
				#predictions = np.reshape(predictions, (predictions.shape[0],predictions.shape[1], 1))
				#self.train_dataset = np.concatenate((self.train_dataset,predictions),axis=2)
				print(self.train_dataset.shape,'dddddddddddddd')
				self.train_labels = save['labels']
				#todo uncomment when use not for submission file
				if(submissionFile == False):
					self.separateValidationAndTestSet()
			del save
		if(submissionFile == False):
			print(self.train_dataset.shape,'train_dataset.shape')
			print(self.train_labels.shape,'train_labels.shape')
			print(self.valid_dataset.shape,'valid_dataset.shape')
			print(self.valid_labels.shape,'valid_labels.shape')
			print(self.test_dataset.shape,'test_dataset.shape')
			print(self.test_labels.shape,'test_labels.shape')

				
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
		
	def prepareDataTolearn(self, longShortTermMemory=False, submissionFile = False):
		self.loadTrainingData(longShortTermMemory, submissionFile)
		if(longShortTermMemory == False):
			self.train_labels = self.convertToOneHot(self.train_labels, self.config.numClasses)
			self.valid_labels = self.convertToOneHot(self.valid_labels, self.config.numClasses)
			self.test_labels = self.convertToOneHot(self.test_labels, self.config.numClasses)

			#Keras has built in function for separating validation and dataset
			#self.train_dataset = np.r_[self.train_dataset , self.valid_dataset]
			#self.train_labels = np.r_[self.train_labels, self.valid_labels]
			#reshape images for alexNet
			
			self.train_dataset /=255
			self.valid_dataset /=255
			self.test_dataset /=255
			'''
			self.train_dataset = np.reshape(self.train_dataset, (len(self.train_dataset[:,1]),self.config.maxHeight*self.config.maxWidth))
			self.test_dataset = np.reshape(self.test_dataset, (len(self.test_dataset[:,1]), self.config.maxHeight*self.config.maxWidth))
			self.valid_dataset = np.reshape(self.valid_dataset, (len(self.valid_dataset[:,1]), self.config.maxHeight* self.config.maxWidth))
			'''
			#print(self.train_labels.shape,'self.train_labels shape')
			#self.separateValidationAndTestSet()
	
		
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
		'''
		print(self.train_dataset.shape,'train_dataset.shape')
		print(self.train_labels.shape,'train_labels.shape')
		print(self.valid_dataset.shape,'valid_dataset.shape')
		print(self.valid_labels.shape,'valid_labels.shape')
		print(self.test_dataset.shape,'test_dataset.shape')
		print(self.test_labels.shape,'test_labels.shape')
		
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
		print(labels.shape,'labels')
		for label in labels:
			print(label,'label')
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
			moreFiles = glob.glob(self.config.generatedImages + '*.' + 'png')
			files = np.r_[files , moreFiles]
			filesNegative = glob.glob(self.config.imgFalseDir + '*.' + fileFormat)
			dataset = np.ndarray(shape=(len(files) + len(filesNegative), height * width),
								dtype=np.float32)
			labels = np.ndarray(shape=(len(files) + len(filesNegative)),dtype=np.int32)
			print(len(files) ,'len(files) ', len(filesNegative),'len(filesNegative)')
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
			
	def generateMoreNervesImages(self, path):
		# load data
		print('ssssudas')

	def generateMoreNervesContoursImages(self):
		'''
		if(os.path.isfile(self.config.pickleName)):
			print('File %s exist, skipping augmentation' % self.config.pickleName)
		else:
		'''
		batchSize = 14
		# load data
		files = glob.glob(self.config.imgNervesContoursDir + '*.tif')
		filesNegative = glob.glob(self.config.imgFalseDir + '*.tif')
		X_train = np.ndarray(shape=(len(files), 1, self.config.maxHeight, self.config.maxWidth),
							dtype=np.float32)
		X_batch = np.ndarray(shape=(200 + 1, 1, self.config.maxHeight, self.config.maxWidth),
							dtype=np.float32)
		y_train = np.ndarray(shape=(len(files)),dtype=np.int32)
		#read data
		num_images = 0
		for image_file in files:
			try:
				image = cv2.imread(image_file, 0)
				#print(mode,'mode',image.shape[0],image.shape[1],'image.shape',height, width,'height * width')
				if(image is not None):
						image = cv2.resize(image,(self.config.maxWidth, self.config.maxHeight), interpolation = cv2.INTER_CUBIC)
						X_train[num_images, ::] = image
						y_train[num_images] = 1
						num_images = num_images + 1
				else:
					print('Could not read: ' + image_file,'image_file')
			except IOError as e:
				print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

		X_train = X_train.astype('float32')
		# define data preparation
		shift = 0.2
		datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, width_shift_range=shift, height_shift_range=shift)
		#datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
		# fit parameters from data
		X_train = X_train.astype('float32')
		datagen.fit(X_train)
		generatedIm = datagen.flow(X_train, y_train, batch_size=batchSize)
		X_batch,y_batch = generatedIm.next()
		#print('X_train',X_batch,len(X_batch),'len X_batch',y_batch,len(y_batch))
		#X_train = X_train.reshape(X_train.shape[0], self.config.maxHeight* self.config.maxWidth)
		X_batch = X_batch.reshape(batchSize, self.config.maxHeight* self.config.maxWidth)
		shuffledData = np.c_[X_batch , y_train]
		np.random.shuffle(shuffledData)
		dataset = shuffledData[:,0 : self.config.maxHeight * self.config.maxWidth]
		labels = shuffledData[:,self.config.maxHeight * self.config.maxWidth ::]
		datasetLenght = dataset.shape[0]		
		#print(dataset.shape,'X_train shape',labels.shape,'X_train')
		'''
		dataset = []
		labels = []
		index_shuf = range(len(X_train.shape[0]))
		shuffle(index_shuf)
		for i in index_shuf:
			dataset.append(X_train[i])
			labels.append(y_train[i])
		'''
		#dataset = X_train
		#labels = y_train
		try:
			f = open(self.config.pickleName, 'wb')
			save = {
				'train_dataset': dataset,
				'train_labels' : labels,
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print('Unable to save data to elis:', e)
			raise
		i = 0
		for generatedImages in dataset:
			oneImage = generatedImages.reshape(self.config.maxHeight, self.config.maxWidth)
			cv2.imwrite(self.config.generatedImages + 'aa_'+ str(i) + '.png', oneImage)	
			i += 1
	def create_submission(self, predictions):
		sub_file = os.path.join('submission_' + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
		subm = open(sub_file, "w")
		subm.write("img,pixels\n")
		overall_mask = cv2.imread('./input/mainMask.tif', cv2.IMREAD_GRAYSCALE)
		overall_mask =  overall_mask.astype(np.uint8)
		for i in range(0, len(self.photosId)):
			subm.write(str(self.photosId[i]) + ',')
			if predictions[i] > -1:
				mask = self.find_best_mask(predictions[i], overall_mask)
				encode = self.rle_encode(mask)
				subm.write(encode)
			subm.write('\n')
		subm.close()
		
	def find_best_mask(self, place, overall_mask):
		'''
		files = glob.glob("./input/trainChanged/*_mask.tif")
		if(not files):
			return 0
		overall_mask = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
		overall_mask.fill(0)
		overall_mask = overall_mask.astype(np.float32)

		for fl in files:
			mask = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
			overall_mask += mask
		overall_mask /= 255
		max_value = overall_mask.max()
		koeff = 0.5
		overall_mask[overall_mask < koeff * max_value] = 0
		overall_mask[overall_mask >= koeff * max_value] = 255
		overall_mask = overall_mask.astype(np.uint8)
		'''

		xFrom = 2 * self.config.xList[place]
		yFrom = 2 * self.config.yList[place]
		fullImage = np.zeros(shape=(self.config.fullImageHeight,self.config.fullImageWidth),dtype=np.uint8)
		fullImage[yFrom :(yFrom +(self.config.maxHeight*2)),
				xFrom :xFrom +(self.config.maxWidth*2)] = overall_mask
		return fullImage
    
	def rle_encode(self, img, order='F'):
		bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
		runs = []
		r = 0
		pos = 1
		for c in bytes:
			if c == 0:
				if r != 0:
					runs.append((pos, r))
					pos += r
					r = 0
				pos += 1
			else:
				r += 1

		if r != 0:
			runs.append((pos, r))
			pos += r

		z = ''
		for rr in runs:
			z += str(rr[0]) + ' ' + str(rr[1]) + ' '
		return z[:-1]
		
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
