import cv2
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import glob
from configFile import Config
from NervesFile import Nerves
from helpers import *
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score,	precision_score, f1_score,roc_curve
#change this number 0- train model, 1 -test model on full images ,2 - visualize learned weghts and activations on good examples
#3 - do it on test set
########################
restore = 1
########################
configModel = Config()
if(restore == 3):
	configModel.trainDir = './input/test/'
	

dataModel = Nerves(configModel)
learning_rate = 0.0006
training_iters = 24000
batch_size = configModel.batch_size
display_step = 10
summaries_dir = '/tmp/nerves_logs9Aug' #tensorboard --logdir='/tmp/nerves_logs9Aug'
# Network Parameters
#n_input = configModel.maxHeight * configModel.maxWidth
n_classes = 2
dropout = 0.75 # Dropout, probability to keep units
validationSetSize = ((training_iters//display_step)//configModel.batch_size)+1

if(restore == 1 or restore == 3):
	#dataModel.loadTrainingDataFullImages(configModel.trainNervesPickle)
	#imageCount = dataModel.train_dataset.shape[0]
	fullImages = dataModel.getAllTrainImgFiles()
	imageCount = len(fullImages)
	print(imageCount,'imageCount')
else:
	dataModel.prepareDataTolearn()
	# I will test in another step
	dataModel.train_dataset = np.concatenate((dataModel.train_dataset,dataModel.test_dataset))
	dataModel.train_labels = np.concatenate((dataModel.train_labels,dataModel.test_labels))
	dataModel.valid_dataset = dataModel.valid_dataset[0:validationSetSize,::]
	dataModel.valid_labels = dataModel.valid_labels[0:validationSetSize]
	dataModel.train_dataset = np.concatenate((dataModel.train_dataset,dataModel.valid_dataset[validationSetSize + 1::,::]))
	dataModel.train_labels = np.concatenate((dataModel.train_labels,dataModel.valid_labels[validationSetSize +1 ::]))
	if(dataModel.train_labels.shape[0]<training_iters):
		training_iters = dataModel.train_labels.shape[0]

#print(dataModel.train_labels.shape,'shapeee',dataModel.train_dataset.shape,type(dataModel.train_labels),type(dataModel.train_dataset))


def getActivations(layer,stimuli):
	'''
	Get activations for certain layer with certain stimuli
	@param tensor - layer for etc convolutional layer with relu activation
	@param image matrix - stimuli as input images
	'''
	print('getActivations')
	units = layer.eval(session=sess,feed_dict={x:np.reshape(stimuli,[1,configModel.maxWidth * configModel.maxHeight],order='F'),keep_prob:1.0})
	plotNNFilter(units,stimuli)

def plotNNFilter(units,stimuli):
	filters = units.shape[3]
	stimuli = np.reshape(stimuli,(configModel.maxHeight, configModel.maxWidth))
	#print(units.shape,'filters shape',filters,stimuli.shape,'stimuli')
	plt.figure(1, figsize=(30,30))
	plt.subplot(4,4,1)
	plt.title('Real image')
	plt.imshow(stimuli, interpolation="nearest", cmap="gray")
	for i in xrange(1,filters):
		plt.subplot(4,(filters//4)+1,i+1)
		plt.title('Filter ' + str(i))
		plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
	plt.show()
	'''
	f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
	ax1.imshow(heatmap0)
	ax2.imshow(heatmap1)
	ax3.imshow(imageMask)
	ax4.imshow(image)
	plt.show()
	'''

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')

	
def fillFeedDict(dataset, labels, step, n_classes, batch_size = 0):
	if(batch_size == 0):
		batch_size = labels.size
	if (labels.shape[0] - batch_size > 0):
		offset = (step* batch_size) % (labels.shape[0]- batch_size)
	else:
		offset = 0
	images = dataset[offset:(offset + batch_size), :]
	label_batch = labels[offset:(offset + batch_size)]
	#print(label_batch.shape,'label_batch shape')
	# Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
	#indices = tf.reshape(tf.range(0, batch_size, 1), [-1, 1])
	#labels_hot = (np.arange(n_classes) == label_batch[:,None]).astype(np.float32)
	return images, label_batch
# Create model
def conv_net1(x, dropout):
	# Store layers weight & bias
	weights = {
		# 5x5 conv, 1 input, 32 outputs
		'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
		# 5x5 conv, 32 inputs, 64 outputs
		'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
		# 5x5 conv, 32 inputs, 64 outputs
		#'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
		# fully connected, 7*7*64 inputs, 1024 outputs
		'wd1': tf.Variable(tf.random_normal([35*30*64, 1024])),
		# 1024 inputs, 10 outputs (class prediction)
		'out': tf.Variable(tf.random_normal([1024, n_classes]))
	}

	biases = {
		'bc1': tf.Variable(tf.random_normal([32])),
		'bc2': tf.Variable(tf.random_normal([64])),
		#'bc3': tf.Variable(tf.random_normal([128])),
		'bd1': tf.Variable(tf.random_normal([1024])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Reshape input picture
	x = tf.reshape(x, shape=[-1,configModel.maxHeight, configModel.maxWidth, 1])
	#print(x.shape,'x shape')
	# Convolution Layer
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)
	# Convolution Layer
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)
	'''
	# Convolution Layer
	conv3 = conv2d(x, weights['wc3'], biases['bc3'])
	# Max Pooling (down-sampling)
	conv3 = maxpool2d(conv3, k=2)
	'''
	# Fully connected layer
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out
# Create model
def conv_net4(x, dropout):
	patch_size = 60
	depth = 16
	num_hidden = 64
	num_hidden2 = 32
	num_channels = 1
	# Reshape input picture
	x = tf.reshape(x, shape=[-1,configModel.maxHeight, configModel.maxWidth, 1])
	
	layer1_weights = tf.Variable(tf.truncated_normal(
	  [patch_size, patch_size, num_channels, depth], stddev=0.1))
	layer1_biases = tf.Variable(tf.zeros([depth]))
	
	layer2_weights = tf.Variable(tf.truncated_normal(
	  [patch_size, patch_size, depth, configModel.maxWidth], stddev=0.1))
	layer2_biases = tf.Variable(tf.constant(1.0, shape=[configModel.maxWidth ]))
	#print(layer1_weights.get_shape(),'layer1_biases')
	#print(layer2_weights.get_shape(),'layer0 biases')
	
	layer3_weights = tf.Variable(tf.truncated_normal(
	  [5, 5 ,configModel.maxWidth, num_hidden], stddev=0.1))#[configModel.maxWidth // 4 * configModel.maxHeight // 4 * depth, num_hidden], stddev=0.1))
	layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	#print(layer3_weights.get_shape(),'layer3_biases')
	
	layer4_weights = tf.Variable(tf.truncated_normal(
	  [17280, num_hidden2], stddev=0.1))
	layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))
	#print(layer4_weights.get_shape(),'layer4_biases')
	layer5_weights = tf.Variable(tf.truncated_normal(
	  [32, configModel.numClasses], stddev=0.1))
	layer5_biases = tf.Variable(tf.constant(1.0, shape=[configModel.numClasses]))
	#print(layer5_weights.get_shape(),'layer5_biases')
	
	#///////////////////////////////////////////////////////////////////
	#layer1
	conv1 = tf.nn.conv2d(x, layer1_weights, [1, 1, 1, 1], padding='SAME')
	hidden1 = tf.tanh(conv1 + layer1_biases)
	pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
						 padding='SAME', name='pool1')
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
					name='norm1')
	#layer2
	conv2 = tf.nn.conv2d(norm1, layer2_weights, [1, 1, 1, 1], padding='SAME')
	hidden2 = tf.tanh(conv2 + layer2_biases)
	pool2 = tf.nn.max_pool(hidden2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],
						 padding='SAME', name='pool1')
	#print(pool2.get_shape(),' pool2',hidden2.get_shape(),'hidden2')
	norm2 = tf.nn.lrn(pool2, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
					name='norm1')

	#layer3
	#print(norm2.get_shape(),' norm2',layer3_weights.get_shape(),'aaaaaaa')
	
	conv3 = tf.nn.conv2d(norm2, layer3_weights, [1, 1, 1, 1], padding='SAME')
	hidden3 = tf.tanh(conv3 + layer3_biases)
	shape = hidden3.get_shape().as_list()
	reshape = tf.reshape(hidden3, [-1, shape[1] * shape[2] * shape[3]])
	
	#print(conv3.get_shape(),'conv3',hidden3.get_shape(),' hidden3',reshape.get_shape(),'reshape')
	#   RELU layer4
	hidden4 = tf.tanh(tf.matmul(reshape, layer4_weights) + layer4_biases)
	#print(hidden4.get_shape(),'hidden4',layer4_biases.get_shape(),' layer4_biases')
	#hidden5 = tf.matmul(hidden4, layer5_weights) + layer4_biases
	
	#layer5
	#hidden = tf.nn.dropout(hidden, keep_prob)
	#print(hidden4.get_shape(),'hidden4',layer5_weights.get_shape(),' layer5_weights',layer5_biases.get_shape(),'layer5_biases')
	result = tf.matmul(hidden4, layer5_weights) + layer5_biases
	layersWeights = {'layer1_weights':layer1_weights,
					'layer2_weights':layer2_weights,
					'layer3_weights':layer3_weights,
					'layer4_weights':layer4_weights,
					'layer5_weights':layer5_weights}
	hiddenLayers = {'hidden1':hidden1,
					'hidden2':hidden2,
					'hidden3':hidden3,
					'hidden4':hidden4}
	return result, layersWeights, hiddenLayers
					
# Create model
def conv_net(x, dropout):
	patch_size = 60
	patch_size2 = 30
	depth = 16
	num_hidden = 64
	num_hidden2 = 32
	num_hidden4 = 28
	num_channels = 1
	# Reshape input picture

	x = tf.reshape(x, shape=[-1,configModel.maxHeight, configModel.maxWidth, 1])
	layer1_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, num_channels, depth], stddev=0.1))
	layer1_biases = tf.Variable(tf.zeros([depth]))
	#print(layer1_weights.get_shape(),'layer1_weights')
	
	
	layer2_weights = tf.Variable(tf.truncated_normal(
			[patch_size2, patch_size2, depth, num_hidden2], stddev=0.1))
	layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2 ]))
	#print(layer2_weights.get_shape(),'layer2_weights')

	
	layer4_weights = tf.Variable(tf.truncated_normal(
			[35*30*32, num_hidden4], stddev=0.1))
	layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden4]))
	#print(layer4_weights.get_shape(),'layer4_weights')
	
	
	layer5_weights = tf.Variable(tf.truncated_normal(
			[num_hidden4, configModel.numClasses], stddev=0.1))
	layer5_biases = tf.Variable(tf.constant(1.0, shape=[configModel.numClasses]))
	#print(layer5_weights.get_shape(),'layer5_weights')
	#///////////////////////////////////////////////////////////////////
	
	#layer1
	conv1 = tf.nn.conv2d(x, layer1_weights, [1, 1, 1, 1], padding='SAME')
	hidden1 = tf.tanh(conv1 + layer1_biases)
	pool1 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
						 padding='SAME', name='pool1')
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75,
					name='norm1')
	#layer2
	conv2 = tf.nn.conv2d(norm1, layer2_weights, [1, 1, 1, 1], padding='SAME')
	hidden2 = tf.tanh(conv2 + layer2_biases)
	pool2 = tf.nn.max_pool(hidden2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1],
						 padding='SAME', name='pool1')
	#print(pool2.get_shape(),' pool2',hidden2.get_shape(),'hidden2')
	norm2 = tf.nn.lrn(pool2, 4, bias=1.0,   alpha=0.001 / 9.0, beta=0.75, name='norm1')
	print(norm2.get_shape(),' norm2')
	
	#layer4 paziurek ieva ar cia isnaudojo shape
	shape = hidden2.get_shape().as_list()
	print(shape,'shape')
	reshape = tf.reshape(hidden2, [-1, shape[1] * shape[2] * shape[3]])
	#reshape = tf.reshape(hidden2, [-1, num_hidden2])
	hidden4 = tf.tanh(tf.matmul(reshape, layer4_weights) + layer4_biases)
	
	print(hidden4.get_shape(),'hidden4',layer5_weights.get_shape(),' layer5_weights',layer5_biases.get_shape(),'layer5_biases')
	#layer5
	result = tf.matmul(hidden4, layer5_weights) + layer5_biases
	print(result.get_shape(),'result')
	
	layersWeights = {'layer1_weights':layer1_weights,
					'layer2_weights':layer2_weights,
					'layer4_weights':layer4_weights,
					'layer5_weights':layer5_weights}
	hiddenLayers = {'hidden1':hidden1,
					'hidden2':hidden2,
					'hidden4':hidden4}
	return result, layersWeights, hiddenLayers
	
graph = tf.Graph()

with graph.as_default():
	# tf Graph input
	x = tf.placeholder(tf.float32, [None, (configModel.maxHeight)* (configModel.maxWidth)])
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	prediction = tf.placeholder(tf.float32, [None, n_classes])
	pred = tf.placeholder(tf.float32, [None, n_classes])
	# Construct model
	pred,filters, hiddenLayers = conv_net(x, keep_prob)

	# Define loss and optimizer
	with tf.name_scope('cost'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
		tf.scalar_summary('cost', cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	
	#filter_summary = tf.image_summary('filter1',filter1)
	#tf.image_summary('filter2',filter2)
	#tf.image_summary('filter3',filter3)
	#tf.image_summary('filter4',filter4)
	#tf.image_summary('filter5',filter5)
	# Evaluate model
	prediction = tf.argmax(pred,1)
	correct_pred = tf.equal(prediction, tf.argmax(y, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		tf.scalar_summary('accuracy', accuracy)

	merged = tf.merge_all_summaries()
	train_writer = tf.train.SummaryWriter(summaries_dir + '/train')
	valid_writer = tf.train.SummaryWriter(summaries_dir + '/valid')
	# Initializing the variables
	init = tf.initialize_all_variables()
	saver = tf.train.Saver()

	# Launch the graph
	with tf.Session() as sess:
		if(restore == 1 or restore == 3):
			saver.restore(sess, configModel.modelWeightsFilePath)	
			#xList=[0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416] 
			#yList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 224, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]		
			slidingWindowCount = len(configModel.xList) #156
			#print(len(xList),'len(xList)',len(yList),'len(yList)')
			#starti moving window on each photo
			stepForWindow = 32
			fullAccuracyArray = np.zeros(shape=(imageCount),dtype=np.float32)
			datasetLSTM = np.zeros(shape=(imageCount,slidingWindowCount,2),dtype=np.float32)
			datasetLSTM2 = np.zeros(shape=(imageCount,slidingWindowCount),dtype=np.float32)
			labelsLSTM = np.zeros(shape=(imageCount,slidingWindowCount,2),dtype=np.float32)
			photoId = []
			for number in range(0, imageCount):
				image = cv2.imread(fullImages[number], 0)
				flbase = os.path.basename(fullImages[number])
				photoId.append(flbase[:-4])
				image = cv2.resize(image,(configModel.fullImageWidth//2, configModel.fullImageHeight//2), interpolation = cv2.INTER_CUBIC)
				if(restore != 3):
					maskImagePath = dataModel.getMaskNameFromImagePath(fullImages[number])
					imageMask = cv2.imread(configModel.trainDir + maskImagePath, 0)
					imageMask = cv2.resize(imageMask,(configModel.fullImageWidth//2, configModel.fullImageHeight//2), interpolation = cv2.INTER_CUBIC)
				batch_dataset = np.ndarray(shape=(slidingWindowCount, (configModel.maxHeight) * (configModel.maxWidth)),dtype=np.float32)
				batch_labels = np.zeros(shape=(slidingWindowCount,2),dtype=np.float32)
				imageNumber = 0
				targetSum = 0
				target = 0
				#xList = []
				#yList = []
				# loop over the image pyramid
				#for resized in pyramid(image, 1.2, (480,400)):
					# loop over the sliding window for each layer of the pyramid
				for (xx, yy, window) in sliding_window(image, stepForWindow, (configModel.maxWidth, configModel.maxHeight)):
					# if the window does not meet our desired window size, ignore it
					if window.shape[0] != configModel.maxHeight or window.shape[1] != configModel.maxWidth:
						continue
					#xList.append(xx)
					#yList.append(yy)
					batch_dataset[imageNumber] = window.reshape(configModel.maxHeight * configModel.maxWidth)
					
					if(restore != 3):
						#findout label for this part of image
						windowMask = imageMask[xx:xx+configModel.maxWidth,yy:yy+configModel.maxHeight]
						#print(np.sum(windowMask[:,:]),'pirma suma', (configModel.maxWidth*configModel.maxHeight)/2.5 ,'turi buti didesne uz sia')
						if(np.sum(windowMask[:,:]) > targetSum):
							target = imageNumber
							targetSum = np.sum(windowMask[:,:])
					imageNumber += 1
				#print('xList',xList,'yList',yList)
				batch_dataset = batch_dataset[:imageNumber-1,::]	
				batch_labels = batch_labels[:imageNumber-1,::]							
				batch_dataset /= 255
				batch_labels[:,0] = 1
				if(restore != 3):	
					if(target > 0):
						#print(number, 'number',target,'imageNumber')
						batch_labels[target,1] = 1
						batch_labels[target,0] = 0

				yfull_train = []
				
				'''
				acc = sess.run([accuracy],feed_dict={x: batch_dataset,
											  y: batch_labels,
											keep_prob: 1.})
				
				#print('Accuracy', acc,len(acc),'acc.shape', fullImages[number],'fullImages[number]')
				fullAccuracyArray[number] = acc[0]
				
				'''
				probabilitiesForClasses = pred.eval(feed_dict={x: batch_dataset,
											  y: batch_labels,
											keep_prob: 1.})

				probabilitiesForClasses1 =  prediction.eval(feed_dict={x: batch_dataset,
											  y: batch_labels,
											keep_prob: 1.})
				datasetLSTM[number,::] = probabilitiesForClasses
				datasetLSTM2[number,::] = probabilitiesForClasses1			
				labelsLSTM[number,::] = batch_labels
				'''
				notNervesClassProbablity = probabilitiesForClasses[:, 0]
				nervesClassProbablity = probabilitiesForClasses[:, 1]
				isNegative0 = False
				isNegative1 = False
				for index, item in enumerate(probabilitiesForClasses):
					if item[0] < 0:
						sNegative0 = True
					if item[1] < 0:
						isNegative1 = True
				
				if(isNegative0):
					notNervesClassProbablity = (notNervesClassProbablity - min(notNervesClassProbablity))*255/max(notNervesClassProbablity)
				else:
					notNervesClassProbablity = notNervesClassProbablity *255/max(notNervesClassProbablity)
					
				if(isNegative1):
					nervesClassProbablity = (nervesClassProbablity - min(nervesClassProbablity))*255/max(nervesClassProbablity)
				else:
					nervesClassProbablity = nervesClassProbablity *255/max(nervesClassProbablity)
				
				#print(nervesClassProbablity.shape,'nervesClassProbablity shape',len(xList),'xList sshape', len(yList), 'yList')
				heatmap00 = np.zeros(shape=(configModel.imageHeight,configModel.imageWidth,1),dtype=np.uint8)
				heatmap11 = np.zeros(shape=(configModel.imageHeight,configModel.imageWidth,1),dtype=np.uint8)

				for number1 in range(0,len(nervesClassProbablity)):
					xxx = number1 % 7
					yyy = round(number1 / 5)
					heatmap00[yyy:yyy+1,xxx:xxx+1,0] = nervesClassProbablity[number1]
					heatmap11[yyy:yyy+1,xxx:xxx+1,0] = notNervesClassProbablity[number1]

				datasetLSTM[number,::] = heatmap00.reshape(heatmap00.shape[0] * heatmap00.shape[1]) #(5,7)
				datasetLSTM2[number,::] = heatmap11.reshape(heatmap11.shape[0] * heatmap11.shape[1]) 
				labelsLSTM[number,::] = batch_labels
				'''
				'''
				cv2.imwrite(configModel.imgClassDir + (dataModel.extractFileName(fullImages[number]))[:-4] + '.png' , heatmap00)
				cv2.imwrite(configModel.imgFalseClassDir + (dataModel.extractFileName(fullImages[number]))[:-4] + '.png' , heatmap11)
				heatmap0 = np.zeros(shape=(configModel.fullImageHeight//2,configModel.fullImageWidth//2,3),dtype=np.uint8)
				heatmap1 = np.zeros(shape=(configModel.fullImageHeight//2,configModel.fullImageWidth//2,3),dtype=np.uint8)

				for number1 in range(0,len(nervesClassProbablity)):
					heatmap0[yList[number1]+ (configModel.maxHeight/2) - stepForWindow:yList[number1] + (configModel.maxHeight/2),
							xList[number1]+ (configModel.maxWidth/2)- stepForWindow:xList[number1] + (configModel.maxWidth/2),  0] = nervesClassProbablity[number1]
					heatmap1[yList[number1]+ (configModel.maxHeight/2) - stepForWindow:yList[number1] + (configModel.maxHeight/2),
							xList[number1]+ (configModel.maxWidth/2)- stepForWindow:xList[number1] + (configModel.maxWidth/2),  0] = notNervesClassProbablity[number1]
								
				f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
				ax1.imshow(heatmap0)
				ax2.imshow(heatmap1)
				ax3.imshow(imageMask)
				ax4.imshow(image)
				plt.show()
				'''
				if(number %20 == 0):
					try:
						f = open(configModel.masksPickle, 'wb')
						save = {
							'dataset': datasetLSTM,
							'dataset2' : datasetLSTM2,
							'labels': labelsLSTM,
							'photoId' : photoId
						}
						pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
						f.close()
					except Exception as e:
						print('Unable to save data to elis:', e)			
						raise
				
				'''
				clone = resized.copy()
				cv2.rectangle(clone, (x, y), (x + configModel.maxWidth, y + configModel.maxHeight), (0, 255, 0), 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(0)
				time.sleep(0.025)
				heatmap0 = np.zeros(shape=(configModel.fullImageHeight,configModel.fullImageWidth,3),dtype=np.uint8)
				heatmap1 = np.zeros(shape=(configModel.fullImageHeight,configModel.fullImageWidth,3),dtype=np.uint8)
				heatmap00 = np.zeros(shape=(configModel.imageHeight,configModel.imageWidth,3),dtype=np.uint8)
				heatmap11 = np.zeros(shape=(configModel.imageHeight,configModel.imageWidth,3),dtype=np.uint8)
				
				for number1 in range(0,len(nervesClassProbablity)):
					heatmap0[yList[number1]+ (configModel.maxHeight/2) - stepForWindow:yList[number1] + (configModel.maxHeight/2),
							xList[number1]+ (configModel.maxWidth/2)- stepForWindow:xList[number1] + (configModel.maxWidth/2),  0] = nervesClassProbablity[number1]
					heatmap1[yList[number1]+ (configModel.maxHeight/2) - stepForWindow:yList[number1] + (configModel.maxHeight/2),
							xList[number1]+ (configModel.maxWidth/2)- stepForWindow:xList[number1] + (configModel.maxWidth/2),  0] = notNervesClassProbablity[number1]
				
				datasetLSTM[number,::] = nervesClassProbablity
				datasetLSTM2[number,::] = notNervesClassProbablity
				labelsLSTM[number,::] = batch_labels
				
				f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
				ax1.imshow(heatmap0)
				ax2.imshow(heatmap1)
				ax3.imshow(imageMask)
				ax4.imshow(image)
				plt.show()
				score = log_loss(batch_labels, predictions_valid)
				print('Score log_loss: ', score)
				#predictions_valid0 = [x[0] for x in predictions_valid]
				predictions_valid1 = [x[1] for x in predictions_valid]
				ind = np.argpartition(predictions_valid1, -4)[-4:]
				for number in range(0,3):
					recY = yList[ind[number]]
					recX = xList[ind[number]]
					cv2.rectangle(image, (recX,recY),
									(recX+configModel.maxWidth,recY+configModel.maxHeight), (0, 255, 0), 2)
				cv2.imshow("Window", image)
				cv2.waitKey(0)
				time.sleep(0.025)
				'''
			'''
			try:
				f = open(configModel.trainNervesLSTMPickle, 'wb')
				save = {
					'datasetLSTM': datasetLSTM,
					'datasetLSTM2' : datasetLSTM2,
					'labelsLSTM': labelsLSTM,
				}
				pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
				f.close()
			except Exception as e:
				print('Unable to save data to elis:', e)			
				raise
			'''
			print(fullAccuracyArray,'fullAccuracyArray',sum(fullAccuracyArray)/len(fullAccuracyArray),'average')
				
		elif(restore == 0):
			sess.run(init)
			step = 1
			print(batch_size,'batch_size')
			# Keep training until reach max iterations
			while step * batch_size < training_iters:
				batch_x, batch_y = fillFeedDict(dataModel.train_dataset, dataModel.train_labels, step, n_classes, batch_size)
				print(batch_x.shape,'batch_x',batch_y.shape,'batch_y')
				# Run optimization op (backprop)
				_, summary = sess.run([optimizer, merged], feed_dict={x: batch_x, y: batch_y,
											   keep_prob: dropout})
				train_writer.add_summary(summary, step)
				if step % display_step == 0:
					# Calculate batch loss and accuracy TODO validation set
					batch_x_valid, batch_y_valid = fillFeedDict(dataModel.valid_dataset, dataModel.valid_labels, step/display_step, n_classes, batch_size)
					loss, acc, summary2, y_pred = sess.run([cost, accuracy, merged, prediction], feed_dict={x: batch_x_valid,
																	  y: batch_y_valid,
																	  keep_prob: 1.})
					print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
						  "{:.6f}".format(loss) + ", Validation Accuracy= " + \
						  "{:.5f}".format(acc)
					saver.save(sess, configModel.modelWeightsFilePath)
					valid_writer.add_summary(summary2, step)
					y_true = np.argmax(batch_y_valid,1)
					print "Precision", precision_score(y_true, y_pred)
					print "Recall", recall_score(y_true, y_pred)
					print "f1_score", f1_score(y_true, y_pred)
					print "confusion_matrix"
					print confusion_matrix(y_true, y_pred)
					fpr, tpr, tresholds = roc_curve(y_true, y_pred)
					print('model saved after iteration steps:', step)
				
				step += 1
			print "Optimization Finished!"
			saver.save(sess, configModel.modelWeightsFilePath)
			print "Model saved!"
			# Calculate accuracy for 256 mnist test images
			
		else:
			saver.restore(sess, configModel.modelWeightsFilePath)
			
			plt.figure(1, figsize=(30,30))
			fromI = 36
			toI = 70
			for i in xrange(fromI,toI):
				plt.subplot(6,6,i+1-fromI)
				plt.title('image ' + str(i))
				image = np.reshape(dataModel.train_dataset[i,::], (configModel.maxHeight, configModel.maxWidth))
				plt.imshow(image, interpolation="nearest", cmap="gray")
			plt.show()
			imageToUse = dataModel.train_dataset[2,::]
			getActivations(hiddenLayers['hidden1'],imageToUse)
			getActivations(hiddenLayers['hidden2'],imageToUse)
			#getActivations(hiddenLayers['hidden3'],imageToUse)
			#getActivations(hiddenLayers['hidden4'],imageToUse)
			
			imageToUse2 = dataModel.train_dataset[27,::]
			getActivations(hiddenLayers['hidden1'],imageToUse2)
			getActivations(hiddenLayers['hidden2'],imageToUse2)
			#getActivations(hiddenLayers['hidden3'],imageToUse2)
			#getActivations(hiddenLayers['hidden4'],imageToUse2)			
			
			imageToUse3 = dataModel.train_dataset[40,::]
			getActivations(hiddenLayers['hidden1'],imageToUse3)
			getActivations(hiddenLayers['hidden2'],imageToUse3)
			#getActivations(hiddenLayers['hidden3'],imageToUse3)
			#getActivations(hiddenLayers['hidden4'],imageToUse3)			
			
			imageToUse4 = dataModel.train_dataset[61,::]
			getActivations(hiddenLayers['hidden1'],imageToUse4)
			getActivations(hiddenLayers['hidden2'],imageToUse4)
			#getActivations(hiddenLayers['hidden3'],imageToUse4)
			#getActivations(hiddenLayers['hidden4'],imageToUse4)
