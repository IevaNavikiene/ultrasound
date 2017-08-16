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
dataModel = Nerves(configModel)
##########################
restore = 0
##########################
training_iters = 5630*35
display_step = 10
validationSetSize = ((training_iters//display_step)//configModel.batch_size)+1

if(restore == 1):
	fullImages = dataModel.getAllTrainImgFiles()
	imageCount = len(fullImages)
	print(imageCount,'imageCount')
else:
	dataModel.prepareDataTolearn(True)
	#dataModel.train_dataset = np.concatenate((dataModel.train_dataset,dataModel.test_dataset))
	#dataModel.train_labels = np.concatenate((dataModel.train_labels,dataModel.test_labels))
	dataModel.valid_dataset = dataModel.valid_dataset[0:validationSetSize,::]
	dataModel.valid_labels = dataModel.valid_labels[0:validationSetSize]
	dataModel.train_dataset = np.concatenate((dataModel.train_dataset,dataModel.valid_dataset[validationSetSize + 1::,::]))
	dataModel.train_labels = np.concatenate((dataModel.train_labels,dataModel.valid_labels[validationSetSize +1 ::]))
	if(dataModel.train_labels.shape[0]<training_iters):
		training_iters = dataModel.train_labels.shape[0]
#print(dataModel.train_labels.shape,'shapeee',dataModel.train_dataset.shape,type(dataModel.train_labels),type(dataModel.train_dataset))

learning_rate = 0.001
training_iters = 10000
batch_size = configModel.batch_size
display_step = 10
summaries_dir = '/tmp/nerves_logs3_LSTM' #tensorboard --logdir='/tmp/nerves_logs3_LSTM'
dropout = 0.75 # Dropout, probability to keep units
num_unrollings=2
num_nodes = 70
n_hidden = 128 # hidden layer num of features

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 3 # MNIST data input (img shape: 28*28)
n_classes = configModel.numClasses # MNIST total classes (0-9 digits)


'''
class BatchGenerator(object):
	def __init__(self, text, batch_size, num_unrollings):
		self._text = text
		self._text_size = len(text)
		self._batch_size = batch_size
		self._num_unrollings = num_unrollings
		segment = self._text_size / batch_size
		self._cursor = [ offset * segment for offset in xrange(batch_size+1)]
		self._last_batch = self._next_batch()
		print('self._last_batch',self._last_batch)
  
	def _next_batch(self):
		"""Generate a single batch from the current cursor position in the data."""
		batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
		textSize = len(self._text)
		for b in xrange(self._batch_size):
			if(self._cursor[b+1] < textSize):
				batch[b, char2id(self._text[self._cursor[b]] + self._text[self._cursor[b+1]])] = 1.0
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



graph = tf.Graph()
with graph.as_default():
  
	# Parameters:
	# Input gate: input, previous output, and bias.
	ifcox = tf.Variable(tf.truncated_normal([vocabulary_size, 4 * num_nodes], -0.1, 0.1))
	ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
	ifcob = tf.Variable(tf.truncated_normal([1, 4 * num_nodes], -0.1, 0.1))
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


'''
def fillFeedDict(dataset, labels, step, test= False):
	'''
	if (labels.shape[0] - batch_size > 0):
		offset = (step* batch_size) % (labels.shape[0]- batch_size)
	else:
		offset = 0
	'''
	#print(dataset.shape[2],'dataset.shape[2]',batch_size,'batch_size')
	images = np.zeros(shape=(labels.shape[1],dataset.shape[2]),dtype=np.float32)
	label_batch = np.zeros(shape=(labels.shape[1],labels.shape[2]),dtype=np.float32)
	images = dataset[step,:,:]
	label_batch = labels[step,:,:]
	tikrinimas = images[:,2] - label_batch[:,1]
	for item in enumerate(tikrinimas):
		if(item == 1 or item == -1):
			print(item,'item')	
	'''
	found = 0
	for i in range(0,34):
		print('parametrai',images[i,::],'atsakymas,',label_batch[i,1])
	for index2, item in enumerate(label_batch):
		if(found ==0 and item[1] == 1):
			found = 1
			#print(item,'item',index2,'ind33ex2')
	'''
	return images, label_batch
# tf Graph input
x = tf.placeholder("float", [None, 3])
y = tf.placeholder("float", [None,configModel.numClasses])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
with tf.name_scope('cost'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	tf.scalar_summary('cost', cost)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
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
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_x, batch_y = fillFeedDict(dataModel.train_dataset, dataModel.train_labels, step)
		#print(batch_x,'batch_x',batch_y,'batch_y')
		# batch_x, batch_y = mnist.train.next_batch(batch_size)
		# Reshape data to get 28 seq of 28 elements
		#batch_x = batch_x.reshape((batch_size, configModel.imageHeight, configModel.imageWidth))
		#batch_x = batch_x.reshape((1, batch_x.shape[0],batch_x.shape[1]))
		#batch_y = batch_y.reshape((1, batch_y.shape[0],batch_y.shape[1]))
		#print(batch_x.shape,'batch_x shape',batch_y.shape,'batch_y')
		# Run optimization op (backprop)
		_, summary = sess.run([optimizer, merged], feed_dict={x: batch_x, y: batch_y})
		train_writer.add_summary(summary, step)
		if step % display_step == 0:
			# Calculate batch accuracy
			acc, summary = sess.run([accuracy, merged], feed_dict={x: batch_x, y: batch_y})
			valid_writer.add_summary(summary, step)
			# Calculate batch loss
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
				  "{:.6f}".format(loss) + ", Training Accuracy= " + \
				  "{:.5f}".format(acc)
		step += 1
	print "Optimization Finished!"
	saver.save(sess, configModel.modelWeightsFilePathLSTM)
	print "Model saved!"
	# Calculate accuracy for 128 mnist test images
	#test_len = 128
	test_data, test_label = fillFeedDict(dataModel.test_dataset, dataModel.test_labels, 0,1)
	#test_data = test_data[:test_len].reshape((-1, configModel.maxHeight, configModel.maxWidth))
	#test_label = test_label[:test_len]
	print "Testing Accuracy:", \
		sess.run(accuracy, feed_dict={x: test_data, y: test_label})

