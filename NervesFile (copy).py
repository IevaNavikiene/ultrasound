from __future__ import absolute_import, division, print_function #needed for methon learnNerves
import numpy as np
from six.moves import cPickle as pickle
import os
import glob
from datetime import datetime
import math
import time
import cv2

#learn nerves
import random
import string

import matplotlib.pyplot as plt
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


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size / batch_size
        self._cursor = [ offset * segment for offset in xrange(batch_size)]
        self._last_batch = self._next_batch()
        print('self._last_batch',self._last_batch)
  
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in xrange(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch
  
    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in xrange(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches















# imu x kordinate pridedu ir atimu po puse plocio...
# iskerpu si kvadrata pagal pixelius is nuotraukos ir is kaukes ir issaugau i atitinkamas papkes

class Nerves:
	def __init__(self):
		self.pickle_file = 'elips.pickle'
		self.trainNervesPickle = 'nerves.pickle'
		self.elipseList = []
		self.filesList = []
		self.maxHeight = 140#115
		self.maxWidth = 120#75
		self.fullImageHeight = 420#115
		self.fullImageWidth = 580#75
		self.maskDir = './input/nervesMasks/'
		self.imgDir = './input/nerves/'
		self.imgFalseDir = './input/notNerves/'
		self.trainDir = './input/validationSet/'
		#self.modelWeightsFilePath = 'model_alexnet-74'#'./modelWeights/model.tfl'
		self.modelWeightsFilePath = 'nerve_weights1.hdf'
		self.defaultCentroidX = 200
		self.defaultCentroidY = 100
		self.partForValidation = 0.25
		self.partForTest = 0.25
		self.numClasses = 2
		self.batch_size = 28
		self.num_batches = 128
		self.nfolds = 3
		self.nbEpoch = 50
		self.randomState = 51
		
	def getElipseData(self):		
		with open(self.pickle_file, 'rb') as f:
			save = pickle.load(f)
			self.elipseList = save['elipseList']
			self.filesList = save['filesList']
			del save

	def extractNerves(self):
		self.getElipseData()
		i= 0
		for xy,centers,something in self.elipseList:
			cropXFrom = xy[0] - (self.maxWidth/2)
			cropXTo = xy[0] + (self.maxWidth/2)
			cropYFrom = xy[1] - (self.maxHeight/2)
			cropYTo = xy[1] + (self.maxHeight/2)
			
			imageMaskbase = self.extractFileName(self.filesList[i])
			
			mask = cv2.imread(self.filesList[i], -1)
			crop_mask = mask[cropYFrom:cropYTo, cropXFrom:cropXTo]
			cv2.imwrite(self.maskDir + imageMaskbase, crop_mask)
			
			imageName = self.getImageNameFromMaskPath(self.filesList[i])
			image = cv2.imread(self.trainDir + imageName)
			crop_img = image[cropYFrom:cropYTo, cropXFrom:cropXTo]
			cv2.imwrite(self.imgDir + imageName, crop_img)
			i = i+1
		
	def extractNotNerves(self):
		files = self.getAllMaskFiles()
		i= 0
		for oneFile in files:
			image = cv2.imread(oneFile, -1)
			empty = self.checkIfMasIsEmpty(image)
			if(empty == True):
				cropXFrom = self.defaultCentroidX - (self.maxWidth/2)
				cropXTo = self.defaultCentroidX + (self.maxWidth/2)
				cropYFrom = self.defaultCentroidY - (self.maxHeight/2)
				cropYTo = self.defaultCentroidY + (self.maxHeight/2)
				
				imageName = self.getImageNameFromMaskPath(oneFile)
				image = cv2.imread(self.trainDir + imageName)
				crop_img = image[cropYFrom:cropYTo, cropXFrom:cropXTo]
				cv2.imwrite(self.imgFalseDir + imageName, crop_img)
				if(i > 2320):
					return True
				i += 1
				
	def loadTrainingData(self):
		with open(self.trainNervesPickle, 'rb') as f:
			save = pickle.load(f)
			self.valid_dataset = save['valid_dataset']
			self.valid_labels = save['valid_labels']
			self.test_dataset = save['test_dataset']
			self.test_labels = save['test_labels']
			#if(save['train_dataset'].shape[0] < self.max_steps):
			#raise Exception('Wrong parameter training_iters of class BaseTensorFlow. Max can be %d', save['train_dataset'].shape[0])

			self.train_dataset = save['train_dataset']#[1:self.max_steps]
			self.train_labels = save['train_labels']#[1:self.max_steps]
			#print ('Full training set', save['train_dataset'].shape)
			del save
		
		print(self.train_dataset.shape,'train_dataset.shape')
		print(self.train_labels.shape,'train_labels.shape')
		print(self.valid_dataset.shape,'valid_dataset.shape')
		print(self.valid_labels.shape,'valid_labels.shape')
		print(self.test_dataset.shape,'test_dataset.shape')
		print(self.test_labels.shape,'test_labels.shape')

				
	def loadTrainingDataFullImages(self):
		with open(self.trainNervesPickle, 'rb') as f:
			save = pickle.load(f)
			#print(save,'saveeee')
			self.train_dataset = save['train_dataset']#[1:self.max_steps]
			self.train_labels = save['train_labels']#[1:self.max_steps]
			del save
		
		print(self.train_dataset.shape,'train_dataset.shape')
		print(self.train_labels.shape,'train_labels.shape')
		
	def prepareDataTolearn(self):
		self.loadTrainingData()

		self.train_labels = self.convertToOneHot(self.train_labels, self.numClasses)
		self.valid_labels = self.convertToOneHot(self.valid_labels, self.numClasses)
		self.test_labels = self.convertToOneHot(self.test_labels, self.numClasses)

		#Keras has built in function for separating validation and dataset
		self.train_dataset = np.r_[self.train_dataset , self.valid_dataset]
		self.train_labels = np.r_[self.train_labels, self.valid_labels]
		#reshape images for alexNet
		self.train_dataset /=255
		#self.valid_dataset /=255
		self.test_dataset /=255
		self.train_dataset = np.reshape(self.train_dataset, (len(self.train_dataset[:,1]), 1,self.maxHeight, self.maxWidth))
		self.test_dataset = np.reshape(self.test_dataset, (len(self.test_dataset[:,1]), 1,self.maxHeight, self.maxWidth))
		#self.valid_dataset = np.reshape(self.valid_dataset, (len(self.valid_dataset[:,1]),1, self.maxHeight, self.maxWidth))
		
	def createModel1(self):
		# Model
		input_layer = tflearn.input_data(shape=[None, self.maxHeight * self.maxWidth], name='input')
		dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
		dense2 = tflearn.fully_connected(dense1, 256, name='dense2')
		softmax = tflearn.fully_connected(dense2, self.numClasses, activation='softmax')
		regression = tflearn.regression(softmax, optimizer='adam',
										learning_rate=0.001,
										loss='categorical_crossentropy')
		return model

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
		
	def trainingDataToPickle(self, width, height, mode = 'nerves'):

		if os.path.exists(self.trainNervesPickle):
			# You may override by setting force=True.
			print('%s already present - Skipping pickling.' % self.trainNervesPickle)
			return []
		print('Pickling %s.' % self.trainNervesPickle)
		
		if(mode == 'nerves'):
			files = glob.glob(self.imgDir + '*.tif')
			filesNegative = glob.glob(self.imgFalseDir + '*.tif')
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
			files = glob.glob(self.trainDir + '*[0-9].tif')
			dataset = np.ndarray(shape=(len(files), height * width),
								dtype=np.float32)
			labels = np.ndarray(shape=(len(files)),dtype=np.int32)
		num_images = 0
		
		#read data
		for image_file in files:
			try:
				image = cv2.imread(image_file, 0)
				#print(mode,'mode',image.shape[0],image.shape[1],'image.shape',height, width,'height * width')
				if(image != None):
					if(mode == 'nerves'):
						dataset[num_images, :] = image.reshape(height * width)
						labels[num_images] = 1
					else:
						flbase = os.path.basename(image_file)
						mask_path = "./input/trainChanged/" + flbase[:-4] + "_mask.tif"
						mask = cv2.imread(mask_path, 0)
						if(mask != None):
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
					if(image != None):
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
			labels = shuffledData[:,self.height * width ::]
			datasetLenght = dataset.shape[0]		
			validIndexTo = datasetLenght * self.partForValidation
			testIndexTo = datasetLenght * self.partForTest
		try:
			'''
			'valid_dataset': dataset[0:validIndexTo,:],
			'valid_labels' : labels[0:validIndexTo],
			'test_dataset': dataset[validIndexTo+1:validIndexTo + testIndexTo,:],
			'test_labels' : labels[validIndexTo+1:validIndexTo + testIndexTo],
			'train_dataset': dataset[validIndexTo + testIndexTo+1::,:],
			'train_labels' : labels[validIndexTo + testIndexTo+1::],
			'''
			f = open(self.trainNervesPickle, 'wb')
			save = {
				'train_dataset': dataset,
				'train_labels' : labels,
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print('Unable to save data to elis:', e)
			raise
				
	def learnNerve3(self):
		""" An example showing how to save/restore models and retrieve weights. """
		import tflearn
		self.prepareDataTolearn()
		model = self.createModel()
		# Define classifier, with model checkpoint (autosave)
		model = tflearn.DNN(regression, checkpoint_path='model.tfl.ckpt')

		# Train model, with model checkpoint every epoch and every 200 training steps.
		model.fit(X, Y, n_epoch=1,
				  validation_set=(self.test_dataset, self.test_labels),
				  show_metric=True,
				  snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
				  snapshot_step=500, # Snapshot (save & evalaute) model every 500 steps.
				  run_id='model_and_weights')


		# ---------------------
		# Save and load a model
		# ---------------------

		# Manually save model
		model.save(self.modelWeightsFilePath)

		# Load a model
		model.load(self.modelWeightsFilePath)

		# Or Load a model from auto-generated checkpoint
		# >> model.load("model.tfl.ckpt-500")

		# Resume training
		model.fit(self.train_dataset,self.train_labels, n_epoch=1,
				  validation_set=(self.test_dataset, self.test_labels),
				  show_metric=True,
				  snapshot_epoch=True,
				  run_id='model_and_weights')


		# ------------------
		# Retrieving weights
		# ------------------

		# Retrieve a layer weights, by layer name:
		dense1_vars = tflearn.variables.get_layer_variables_by_name('dense1')
		# Get a variable's value, using model `get_weights` method:
		'''
		print("Dense1 layer weights:")
		print(model.get_weights(dense1_vars[0]))
		# Or using generic tflearn function:
		print("Dense1 layer biases:")
		with model.session.as_default():
			print(tflearn.variables.get_value(dense1_vars[1]))
		
		# It is also possible to retrieve a layer weights throught its attributes `W`
		# and `b` (if available).
		# Get variable's value, using model `get_weights` method:
		print("Dense2 layer weights:")
		print(model.get_weights(dense2.W))
		# Or using generic tflearn function:
		print("Dense2 layer biases:")
		with model.session.as_default():
			print(tflearn.variables.get_value(dense2.b))
		'''
		
	def loadModelAndRunTest(self):
		import tflearn
		from tflearn.layers.core import input_data, dropout, fully_connected
		from tflearn.layers.conv import conv_2d, max_pool_2d
		from tflearn.layers.normalization import local_response_normalization
		from tflearn.layers.estimator import regression

		self.loadTrainingData()
		self.train_labels = self.convertToOneHot(self.train_labels, self.numClasses)
		self.valid_labels = self.convertToOneHot(self.valid_labels, self.numClasses)
		self.test_labels = self.convertToOneHot(self.test_labels, self.numClasses)

		X = np.r_[self.train_dataset , self.valid_dataset]
		Y = np.r_[self.train_labels, self.valid_labels]
		
		#reshape images for alexNet
		X = np.reshape(X, (len(X[:,1]), self.maxHeight, self.maxWidth,1))
		self.test_dataset = np.reshape(self.test_dataset, (len(self.test_dataset[:,1]), self.maxHeight, self.maxWidth, 1))

		# Model
		input_layer = tflearn.input_data(shape=[None, self.maxHeight * self.maxWidth], name='input')
		dense1 = tflearn.fully_connected(input_layer, 128, name='dense1')
		dense2 = tflearn.fully_connected(dense1, 256, name='dense2')
		softmax = tflearn.fully_connected(dense2, self.numClasses, activation='softmax')
		regression = tflearn.regression(softmax, optimizer='adam',
										learning_rate=0.001,
										loss='categorical_crossentropy')

		# Define classifier, with model checkpoint (autosave)
		model = tflearn.DNN(regression, checkpoint_path=self.modelWeightsFilePath)
		model.load(self.modelWeightsFilePath)
		model.fit(X, Y, n_epoch=1,
			  validation_set=(self.test_dataset, self.test_labels),
			  show_metric=True,
			  snapshot_epoch=True,
			  run_id='model_and_weights')	
		
	def extractFileName(self, path):
		imagebase = os.path.basename(path)
		return imagebase

	def getImageNameFromMaskPath(self, path):
		base = self.extractFileName(path)
		imageName = base[:-9] + '.tif'
		return imageName
		
	def getAllMaskFiles(self):
		return glob.glob(self.trainDir + "*_mask.tif")
		
	def getAllTrainImgFiles(self):
		return glob.glob(self.trainDir + "*[0-9].tif")
		
	def checkIfMasIsEmpty(self, mask):
		return np.sum(mask[:,:]) == 0
			
	def createModel(self): #alexnet tfLearn	
		self.loadTrainingData()
		self.train_labels = self.convertToOneHot(self.train_labels, self.numClasses)
		self.valid_labels = self.convertToOneHot(self.valid_labels, self.numClasses)
		self.test_labels = self.convertToOneHot(self.test_labels, self.numClasses)
		
		#X =  np.r_[self.train_dataset , self.valid_dataset]
		#Y = np.r_[self.train_labels, self.valid_labels]
		X = self.train_dataset
		Y = self.train_labels
		#reshape images for alexNet
		X = np.reshape(X, (len(X[:,1]), self.maxHeight, self.maxWidth,1))
		self.test_dataset = np.reshape(self.test_dataset, (len(self.test_dataset[:,1]), self.maxHeight, self.maxWidth, 1))
		self.valid_dataset = np.reshape(self.valid_dataset, (len(self.valid_dataset[:,1]), self.maxHeight, self.maxWidth, 1))
		
		# Building 'AlexNet'
		network = input_data(shape=[None, self.maxHeight, self.maxWidth, 1])
		network = conv_2d(network, 96, 11, strides=4, activation='relu')
		network = max_pool_2d(network, 3, strides=2)
		network = local_response_normalization(network)
		network = conv_2d(network, 256, 5, activation='relu')
		network = max_pool_2d(network, 3, strides=2)
		network = local_response_normalization(network)
		network = conv_2d(network, 384, 3, activation='relu')
		network = conv_2d(network, 384, 3, activation='relu')
		network = conv_2d(network, 256, 3, activation='relu')
		network = max_pool_2d(network, 3, strides=2)
		network = local_response_normalization(network)
		network = fully_connected(network, 4096, activation='tanh')
		network = dropout(network, 0.5)
		network = fully_connected(network, 4096, activation='tanh')
		network = dropout(network, 0.5)
		network = fully_connected(network, self.numClasses, activation='softmax')
		network = regression(network, optimizer='momentum',
							 loss='categorical_crossentropy',
							 learning_rate=0.001)

		# Training
		model = tflearn.DNN(network, checkpoint_path='model_alexnet',
							max_checkpoints=1, tensorboard_verbose=2)
		return model
		# Train model, with model checkpoint every epoch and every 200 training steps.
		model.fit(X, Y, n_epoch=1,
				  validation_set=(self.valid_dataset, self.valid_labels),
				  show_metric=True,
				  snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
				  snapshot_step=500, # Snapshot (save & evalaute) model every 500 steps.
				  run_id='model_and_weights')


		# ---------------------
		# Save and load a model
		# ---------------------
		print('Saving model to : ', self.modelWeightsFilePath)
		model.save(self.modelWeightsFilePath)
		print('Load saved model:')
		model.load(self.modelWeightsFilePath)
		model.fit(X, Y, n_epoch=1,
				  validation_set=(self.test_dataset, self.test_labels),
				  show_metric=True,
				  snapshot_epoch=True,
				  run_id='model_and_weights')

	def print_activations(self,t):
	  print(t.op.name, ' ', t.get_shape().as_list())


	def inference(self,images):
		"""Build the AlexNet model.
		Args:
		images: Images Tensor
		Returns:
		pool5: the last Tensor in the convolutional component of AlexNet.
		parameters: a list of Tensors corresponding to the weights and biases of the
			AlexNet model.
		"""
		parameters = []
		# conv1
		with tf.name_scope('conv1') as scope:
			kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
								 trainable=True, name='biases')
			bias = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(bias, name=scope)
			self.print_activations(conv1)
			parameters += [kernel, biases]

		# lrn1
		# TODO(shlens, jiayq): Add a GPU version of local response normalization.

		# pool1
		pool1 = tf.nn.max_pool(conv1,
							 ksize=[1, 3, 3, 1],
							 strides=[1, 2, 2, 1],
							 padding='VALID',
							 name='pool1')
		self.print_activations(pool1)

		# conv2
		with tf.name_scope('conv2') as scope:
			kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
								 trainable=True, name='biases')
			bias = tf.nn.bias_add(conv, biases)
			conv2 = tf.nn.relu(bias, name=scope)
			parameters += [kernel, biases]
		self.print_activations(conv2)

		# pool2
		pool2 = tf.nn.max_pool(conv2,
							 ksize=[1, 3, 3, 1],
							 strides=[1, 2, 2, 1],
							 padding='VALID',
							 name='pool2')
		self.print_activations(pool2)

		# conv3
		with tf.name_scope('conv3') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
													 dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
								 trainable=True, name='biases')
			bias = tf.nn.bias_add(conv, biases)
			conv3 = tf.nn.relu(bias, name=scope)
			parameters += [kernel, biases]
			self.print_activations(conv3)

		# conv4
		with tf.name_scope('conv4') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
													 dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								 trainable=True, name='biases')
			bias = tf.nn.bias_add(conv, biases)
			conv4 = tf.nn.relu(bias, name=scope)
			parameters += [kernel, biases]
			self.print_activations(conv4)

		# conv5
		with tf.name_scope('conv5') as scope:
			kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
													 dtype=tf.float32,
													 stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
								 trainable=True, name='biases')
			bias = tf.nn.bias_add(conv, biases)
			conv5 = tf.nn.relu(bias, name=scope)
			parameters += [kernel, biases]
			self.print_activations(conv5)

		# pool5
		pool5 = tf.nn.max_pool(conv5,
							 ksize=[1, 3, 3, 1],
							 strides=[1, 2, 2, 1],
							 padding='VALID',
							 name='pool5')
		self.print_activations(pool5)

		return pool5, parameters


	def time_tensorflow_run(self,session, target, info_string):
		"""Run the computation to obtain the target tensor and print timing stats.
		Args:
		session: the TensorFlow session to run the computation under.
		target: the target Tensor that is passed to the session's run() function.
		info_string: a string summarizing this run, to be printed with the stats.
		Returns:
		None
		"""
		num_steps_burn_in = 10
		total_duration = 0.0
		total_duration_squared = 0.0
		for i in xrange(self.num_batches + num_steps_burn_in):
			start_time = time.time()
			_ = session.run(target)
			duration = time.time() - start_time
			if i > num_steps_burn_in:
			  if not i % 10:
				print ('%s: step %d, duration = %.3f' %
						 (datetime.now(), i - num_steps_burn_in, duration))
				total_duration += duration
				total_duration_squared += duration * duration
		mn = total_duration / self.num_batches
		vr = total_duration_squared / self.num_batches - mn * mn
		sd = math.sqrt(vr)
		print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
			 (datetime.now(), info_string, self.num_batches, mn, sd))


	def learnNerve2(self):
		#reshape images for alexNet
		self.train_dataset = np.reshape(self.train_dataset, (len(self.train_dataset[:,1]), self.maxHeight, self.maxWidth,1))
		self.test_dataset = np.reshape(self.test_dataset, (len(self.test_dataset[:,1]), self.maxHeight, self.maxWidth, 1))
		self.valid_dataset = np.reshape(self.valid_dataset, (len(self.valid_dataset[:,1]), self.maxHeight, self.maxWidth, 1))
		"""Run the benchmark on AlexNet."""
		with tf.Graph().as_default():
			# Note that our padding definition is slightly different the cuda-convnet.
			# In order to force the model to start with the same activations sizes,
			# we add 3 to the image_size and employ VALID padding above.
			images = tf.Variable(tf.random_normal([self.batch_size,
													 self.maxHeight,
													 self.maxWidth, 1],
													dtype=tf.float32,
													stddev=1e-1))

			# Build a Graph that computes the logits predictions from the
			# inference model.
			pool5, parameters = self.inference(images)

			# Build an initialization operation.
			init = tf.initialize_all_variables()

			# Start running operations on the Graph.
			config = tf.ConfigProto()
			config.gpu_options.allocator_type = 'BFC'
			sess = tf.Session(config=config)
			sess.run(init)

			# Run the forward benchmark.
			self.time_tensorflow_run(sess, pool5, "Forward")

			# Add a simple objective so we can calculate the backward pass.
			objective = tf.nn.l2_loss(pool5)
			# Compute the gradient with respect to all the parameters.
			grad = tf.gradients(objective, parameters)
			# Run the backward benchmark.
			self.time_tensorflow_run(sess, grad, "Forward-backward")
	'''		
	def createModel(self):
		model = Sequential()
		model.add(Convolution2D(4, 1, 3, border_mode='same', init='he_normal',
								input_shape=(1, self.maxHeight, self.maxWidth)))

		model.add(MaxPooling2D(pool_size=(2, 2)))	
		model.add(Dropout(0.2))
		model.add(Convolution2D(8, 1, 3, border_mode='same', init='he_normal'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.2))

		#model.add(Convolution2D(8, 4, 4, border_mode='same', init='he_normal'))
		#model.add(Convolution2D(8, 6, 6, border_mode='same', init='he_normal'))
		#model.add(Convolution2D(8, 6, 6, border_mode='same', init='he_normal'))

		model.add(Flatten())
		#model.add(Dense(2))
		model.add(Dense(2, W_regularizer=l2(0.008), activity_regularizer=activity_l2(0.01)))
		model.add(Activation('softmax'))

		sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd, loss='categorical_crossentropy')
		return model
	'''	
	
	def learnNerve(self):
		self.prepareDataTolearn()

		yfull_train = dict()
		yfull_test = []
		kf = KFold(len(self.train_dataset), n_folds=self.nfolds, shuffle=True, random_state=self.randomState)
	def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (mostl likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    #print('batchesbatchesaaaaaaaaaaaaaa',batches)
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

	train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
	valid_batches = BatchGenerator(valid_text, 1, 1)

	print batches2string(train_batches.next())
	print batches2string(train_batches.next())
	print batches2string(valid_batches.next())
	print batches2string(valid_batches.next())

	def logprob(predictions, labels):
		"""Log-probability of the true labels in a predicted batch."""
		predictions[predictions < 1e-10] = 1e-10
		return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

	def sample_distribution(distribution):
		"""Sample one element from a distribution assumed to be an array of normalized
		probabilities.
		"""
		r = random.uniform(0, 1)
		s = 0
		for i in xrange(len(distribution)):
			s += distribution[i]
			if s >= r:
				return i
		return len(distribution) - 1

	def sample(prediction):
		"""Turn a (column) prediction into 1-hot encoded samples."""
		p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
		p[0, sample_distribution(prediction[0])] = 1.0
		return p

	def random_distribution():
		"""Generate a random column of probabilities."""
		b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
		return b/np.sum(b, 1)[:,None]

	def bigrams(embeddings):
		"""Turn a probability distribution over the bigrams into the most likely
		   bigram string representations."""
		bigrams = []
		for i in range(embeddings.shape[0]):
			cosims = np.dot(final_embeddings, embeddings[i]) / la.norm(embeddings[i])
			bigrams.append(reverse_bigram_dict[np.argmax(cosims)])
		return " ".join(bigrams)

	num_nodes = 64

	graph = tf.Graph()
	with graph.as_default():
	  
		# Parameters:
		# Input gate: input, previous output, and bias.
		ifcox = tf.Variable(tf.truncated_normal([vocabulary_size, 4 * num_nodes], -0.1, 0.1))
		ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
		ifcob = tf.Variable(tf.truncated_normal([1, 4 * num_nodes], -0.1, 0.1))
	   
		'''
		ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1)) #why meant is not on zero axis?
		#tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
		im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		ib = tf.Variable(tf.zeros([1, num_nodes]))
		# Forget gate: input, previous output, and bias.
		fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		fb = tf.Variable(tf.zeros([1, num_nodes]))
		# Memory cell: input, state and bias.                             
		cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		cb = tf.Variable(tf.zeros([1, num_nodes]))
		# Output gate: input, previous output, and bias.
		ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
		om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
		ob = tf.Variable(tf.zeros([1, num_nodes]))
		'''
		# Variables saving state across unrollings.
		saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
		saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
		# Classifier weights and biases.
		w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
		b = tf.Variable(tf.zeros([vocabulary_size]))


		# Definition of the cell computation.
		def lstm_cell(i, o, state):
			"""Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
			Note that in this formulation, we omit the various connections between the
			previous state and the gates."""
			'''
			print(i,'eeeeeeeeeiiiiiiii')
			print(o.eval,'output')
			print(ix.eval,'ix')
			print(im.eval,'im')
			print(ib.eval,'ib')
			full1= tf.matmul(i, ix)
			print(full1)
			full2= tf.matmul(o, im)
			print(full2)
			full3=full1+full2+ib
			print(full3)
			input_gate = tf.sigmoid(full3)
			
			input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
			forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
			update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
			state = forget_gate * state + input_gate * tf.tanh(update)
			output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
			
			print(i,'eeeeeeeeeiiiiiiii')
			print(o,'output')
			print(ifcox,'ifcox')
			print(ifcom,'ifcom')
			print(ifcob,'ifcob')
			'''
			all_gates_state =  tf.matmul(i, ifcox) + tf.matmul(o, ifcom)+ ifcob
			input_gate = tf.sigmoid(all_gates_state[:, 0:num_nodes])
			forget_gate = tf.sigmoid(all_gates_state[:, num_nodes: 2*num_nodes])
			update = all_gates_state[:, 2*num_nodes: 3*num_nodes]
			state = forget_gate * state + input_gate * tf.tanh(update)
			output_gate = tf.sigmoid(all_gates_state[:, 3*num_nodes:])
			return output_gate * tf.tanh(state), state

		# Input data.
		train_data = list()
		for _ in xrange(num_unrollings + 1):
			train_data.append(
			  tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
		train_inputs = train_data[:num_unrollings]
		train_labels = train_data[1:]  # labels are inputs shifted by one time step.

		# Unrolled LSTM loop.
		outputs = list()
		output = saved_output
		state = saved_state
		for i in train_inputs:
			output, state = lstm_cell(i, output, state)
			outputs.append(output)

		# State saving across unrollings.
		with tf.control_dependencies([saved_output.assign(output),
									saved_state.assign(state)]):
			# Classifier.
			logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
			loss = tf.reduce_mean(
			  tf.nn.softmax_cross_entropy_with_logits(
				logits, tf.concat(0, train_labels)))
			tf.scalar_summary('cross entropy', loss)

		# Optimizer.
		global_step = tf.Variable(0)
		learning_rate = tf.train.exponential_decay(
		  10.0, global_step, 5000, 0.1, staircase=True)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		gradients, v = zip(*optimizer.compute_gradients(loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
		optimizer = optimizer.apply_gradients(
		  zip(gradients, v), global_step=global_step)

		# Predictions.
		train_prediction = tf.nn.softmax(logits)
	  
		# Sampling and validation eval: batch 1, no unrolling.
		sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
		saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
		saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
		reset_sample_state = tf.group(
		  saved_sample_output.assign(tf.zeros([1, num_nodes])),
		  saved_sample_state.assign(tf.zeros([1, num_nodes])))
		sample_output, sample_state = lstm_cell(
		  sample_input, saved_sample_output, saved_sample_state)
		with tf.control_dependencies([saved_sample_output.assign(sample_output),
									saved_sample_state.assign(sample_state)]):
			sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))
			
	num_steps = 7001
	summary_frequency = 100

	with tf.Session(graph=graph) as session:
		tf.initialize_all_variables().run()
		merged_summary_op = tf.merge_all_summaries()
		summary_writer = tf.train.SummaryWriter(summaries_dir, graph=tf.get_default_graph())
		print 'Initialized'
		mean_loss = 0
		for step in xrange(num_steps):
			batches = train_batches.next()
			feed_dict = dict()
			for i in xrange(num_unrollings + 1):
				feed_dict[train_data[i]] = batches[i]
			_, l, predictions, lr, summary = session.run(
			  [optimizer, loss, train_prediction, learning_rate, merged_summary_op], feed_dict=feed_dict)
			#write everything to summary
			summary_writer.add_summary(summary, step)
			summary_writer.flush()
			mean_loss += l
			if step % summary_frequency == 0:
				if step > 0:
					mean_loss = mean_loss / summary_frequency
				# The mean loss is an estimate of the loss over the last few batches.
				print 'Average loss at step', step, ':', mean_loss, 'learning rate:', lr
				mean_loss = 0
				labels = np.concatenate(list(batches)[1:])
				print 'Minibatch perplexity: %.2f' % float(
				  np.exp(logprob(predictions, labels)))
				if step % (summary_frequency * 10) == 0:
					# Generate some samples.
					print '=' * 80
					for _ in xrange(5):
						feed = sample(random_distribution())
						sentence = characters(feed)[0]
						reset_sample_state.run()
						for _ in xrange(79):
							prediction = sample_prediction.eval({sample_input: feed})
							feed = sample(prediction)
							sentence += characters(feed)[0]
						print sentence
					print '=' * 80
				# Measure validation set perplexity.
				reset_sample_state.run()
				valid_logprob = 0
				for _ in xrange(valid_size):
					b = valid_batches.next()
					predictions = sample_prediction.eval({sample_input: b[0]})
					valid_logprob = valid_logprob + logprob(predictions, b[1])
				print 'Validation set perplexity: %.2f' % float(np.exp(
				  valid_logprob / valid_size))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	def learnNerve4(self):
		self.prepareDataTolearn()

		yfull_train = dict()
		yfull_test = []
		kf = KFold(len(self.train_dataset), n_folds=self.nfolds, shuffle=True, random_state=self.randomState)
		for train_index, test_index in kf:
			self.train_dataset_batch, self.valid_dataset_batch = self.train_dataset[train_index], self.train_dataset[test_index]
			self.train_labels_batch, self.valid_labels_batch = self.train_labels[train_index], self.train_labels[test_index]
			model = self.createModel()

			callbacks = [
				#EarlyStopping(monitor='val_loss', patience=2, verbose=0),
			]
			model.fit(self.train_dataset_batch, self.train_labels_batch, batch_size=self.batch_size, nb_epoch=self.nbEpoch,
				  shuffle=True, verbose=2, validation_data=(self.valid_dataset, self.valid_labels),
				  callbacks=callbacks)
				  
			#save weights--------
			model.save_weights('a' + str(i)+ self.modelWeightsFilePath+ str(i))
			'''
			if True and os.path.exists(WEIGHTS_FNAME):
				# Just change the True to false to force re-training
				print('Loading existing weights')
				model.load_weights(WEIGHTS_FNAME+ str(i)+ '.hdf')
			else:
				batch_size = 128
				self.nbEpoch = 12
				model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=self.nbEpoch,
						  show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
				model.save_weights(WEIGHTS_FNAME+ str(i)+ '.hdf')
			'''
			predictions_valid = model.predict(self.valid_dataset_batch, batch_size=batch_size, verbose=1)
			score = log_loss(self.valid_labels_batch, predictions_valid)
			print('Score log_loss: ', score)

			# Store valid predictions
			for i in range(len(test_index)):
				yfull_train[test_index[i]] = predictions_valid[i]

			# Store test predictions
			test_prediction = model.predict(self.test_dataset, batch_size=self.batch_size, verbose=2)
			yfull_test.append(test_prediction)

		predictions_valid = self.get_validation_predictions(self.train_dataset, yfull_train)
		score = log_loss(self.train_labels, predictions_valid)
		print("Log_loss train independent avg: ", score)

		print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, self.maxHeight, self.maxWidth, self.nfolds, self.nbEpoch))
		perc = self.getPredScorePercent(self.train_labels, train_id, predictions_valid)
		print('Percent success: {}'.format(perc))

		info_string = 'loss_' + str(score) \
						+ '_r_' + str(self.maxHeight) \
						+ '_c_' + str(self.maxWidth) \
						+ '_folds_' + str(self.nfolds) \
						+ '_ep_' + str(self.nbEpoch)

		test_res = self.merge_several_folds_mean(yfull_test, self.nfolds)
		#create_submission(test_res, test_id, info_string)
 
	def merge_several_folds_mean(data, nfolds):
		a = np.array(data[0])
		for i in range(1, nfolds):
			a += np.array(data[i])
		a /= nfolds
		return a.tolist()

	def getPredScorePercent(train_target, train_id, predictions_valid):
		perc = 0
		for i in range(len(train_target)):
			pred = 1
			if predictions_valid[i][0] > 0.5:
				pred = 0
			real = 1
			if train_target[i][0] > 0.5:
				real = 0
			if real == pred:
				perc += 1
		perc /= len(train_target)
		return perc
	#is pradziu pratestuoju kaip dabar veikia modelis, o po to imsiuosi antrojo apmokymo?
		
	def testWeightsOnRealImages(self):
		self.loadTrainingDataFullImages()
		'''
		#Keras has built in function for separating validation and dataset
		self.train_dataset = np.r_[self.train_dataset , self.valid_dataset]
		self.train_labels = np.r_[self.train_labels, self.valid_labels]
		'''
		#reshape images
		self.train_dataset /=255
		imageCount = self.train_dataset.shape[0]
		#self.train_dataset = np.reshape(self.train_dataset, (len(self.train_dataset[:,1]), 1,self.fullImageHeight, self.fullImageWidth))

		model = self.createModel()

		if True and os.path.exists(self.modelWeightsFilePath):
			# Just change the True to false to force re-training
			print('Loading existing weights')
			model.load_weights(self.modelWeightsFilePath)
		else:
			print('Weights file does not exist : %s ' %  self.modelWeightsFilePath)
		slidingWindowCount = 156
		xList=[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416] 
		yList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]		
		#starti moving window on each photo
		for number in range(0, imageCount):
			image = self.train_dataset[number,:]
			image = np.reshape(image, (self.fullImageHeight, self.fullImageWidth, 1))
			batch_dataset = np.ndarray(shape=(slidingWindowCount, 1, self.maxHeight, self.maxWidth),dtype=np.float32)
			batch_labels = np.zeros(shape=(slidingWindowCount),dtype=np.int32)
			imageNumber = 0
			# loop over the image pyramid
			for resized in self.pyramid(image, 1.2, (480,400)):
				# loop over the sliding window for each layer of the pyramid
				for (x, y, window) in self.sliding_window(resized, 32, (self.maxWidth, self.maxHeight)):
					# if the window does not meet our desired window size, ignore it
					if window.shape[0] != self.maxHeight or window.shape[1] != self.maxWidth:
						continue
					batch_dataset[imageNumber] = np.reshape(window, 
														(1, self.maxHeight, self.maxWidth))
					imageNumber += 1
					'''
					clone = resized.copy()
					cv2.rectangle(clone, (x, y), (x + self.maxWidth, y + self.maxHeight), (0, 255, 0), 2)
					cv2.imshow("Window", clone)
					cv2.waitKey(0)
					time.sleep(0.025)
					'''
										
					
			#39 is random picked number
			batch_labels[39] = 1
			yfull_train = []
			predictions_valid = model.predict(batch_dataset, batch_size=imageNumber, verbose=1)
			score = log_loss(batch_labels, predictions_valid)
			print('Score log_loss: ', score)
			#predictions_valid0 = [x[0] for x in predictions_valid]
			predictions_valid1 = [x[1] for x in predictions_valid]
			ind = np.argpartition(predictions_valid1, -4)[-4:]
			for number in range(0,3):
				recY = yList[ind[number]]
				recX = xList[ind[number]]
				cv2.rectangle(image, (recX,recY),
								(recX+self.maxWidth,recY+self.maxHeight), (0, 255, 0), 2)
			cv2.imshow("Window", image)
			cv2.waitKey(0)
			time.sleep(0.025)
		
		'''

		# Store valid predictions
		for i in range(len(test_index)):
			yfull_train[test_index[i]] = predictions_valid[i]

		predictions_valid = self.get_validation_predictions(self.train_dataset, yfull_train)
		score = log_loss(self.train_labels, predictions_valid)
		print('Final log_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(score, self.nfolds, self.nbEpoch))
		perc = self.getPredScorePercent(self.train_labels, train_id, predictions_valid)
		print('Percent success: {}'.format(perc))

		info_string = 'loss_' + str(score) \
						+ '_r_' + str(self.maxHeight) \
						+ '_c_' + str(self.maxWidth) \
						+ '_folds_' + str(self.nfolds) \
						+ '_ep_' + str(self.nbEpoch)
		'''				
