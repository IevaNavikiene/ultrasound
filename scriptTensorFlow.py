#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
#ka liko padaryti:
# pickle failiuke kolkas paveiksliukas issaugotas kaip 32ant32 , o ne 32*32 
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
__author__ = 'ievava: https://kaggle.com/ievava'
np.random.seed(212)
import cv2
import os
import glob
import datetime
import time
from sklearn.cross_validation import KFold
from keras.utils import np_utils
from keras.utils import np_utils
from six.moves import cPickle as pickle
#from sklearn.metrics import log_loss
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 50, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('training_data_dir', './input/trainChanged2/', 'Directory for storing data')
flags.DEFINE_string('test_data_dir', './input/test2/', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/ultrasound_logs', 'Summaries directory')
flags.DEFINE_string('batch_size', 5, '')
flags.DEFINE_string('image_size', 32, '')
flags.DEFINE_string('num_classes', 2, '')
flags.DEFINE_string('depth', 16, '')
pickle_file = 'ultrasound.pickle'
#load train images with masks
def load_train(img_rows, img_cols):
    X_train = []
    X_train_id = []
    mask_train = []
    start_time = time.time()

    print('Read train images')
    files = glob.glob(FLAGS.training_data_dir + "*[0-9].tif")
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl, FLAGS.image_size, FLAGS.image_size)
        X_train.append(img)
        X_train_id.append(flbase[:-4])
        mask_path = FLAGS.training_data_dir + flbase[:-4] + "_mask.tif"
        mask = get_im_cv2(mask_path, FLAGS.image_size, FLAGS.image_size)
        mask_train.append(mask)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, mask_train, X_train_id
            
def read_and_normalize_train_data(img_rows, img_cols):
    with tf.name_scope('input'):
        train_dataset, train_labels, train_id = load_train(img_rows, img_cols)
        
        train_dataset = train_dataset.reshape((-1, img_rows * img_cols)).astype(np.float32)
        tf.image_summary('input', train_dataset, 10)
        print ('Training set', train_dataset.shape, train_labels.shape)
        return train_dataset, train_labels, train_id
        '''
        train_data = np.array(train_data, dtype=np.uint8)
        train_target = np.array(train_target, dtype=np.uint8)
        print('train_data.shape[0]',train_data.shape)
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
        # Convert to 0 or 1
        train_target = get_empty_mask_state(train_target)
        train_target = np_utils.to_categorical(train_target, 2)
        train_data = train_data.astype('float32')
        train_data /= 255
        print('Train shape:', train_data.shape)
        print(train_data.shape[0], 'train samples')
        return train_data, train_target, train_id
        '''

       
#what is done with masks ka sitas daro? kazkaip atrenka geriausias mask?
def find_best_mask():
    files = glob.glob(IMAGES_TRAIN_PATH + "*_mask.tif")
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
    #print('overall_mask',overall_mask)
    return overall_mask

#convert mask into type proper for submission 
def rle_encode(img, order='F'):
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
  
#creaate submission file with masks for test set
#??? what is in prediction matrix
def create_submission(predictions, test_id, info):
    sub_file = os.path.join('submission_' + info + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
    subm = open(sub_file, "w")
    mask = find_best_mask()
    encode = rle_encode(mask)
    subm.write("img,pixels\n")
    print(predictions,'predictions')
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        if predictions[i][1] > 0.5:
            subm.write(encode)
        subm.write('\n')
    subm.close()
        
        
        
        
        
        
        
#other algorithm        
def dense_to_one_hot(labels_dense, num_classes=10):
  labels_dense = np.array(labels_dense)
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(FLAGS.num_classes) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

# This is our model, a very simple, 1-layer MLP
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights[0]), _biases[0]))
    return tf.matmul(layer_1, _weights[1]) + _biases[1]

def train():
  with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      #print('sssacve',save)
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']

      if(save['train_dataset'].shape[0] < FLAGS.max_steps):
            raise Exception('Wrong parameter training_iters of class BaseTensorFlow. Max can be %d', save['train_dataset'].shape[0])

      train_dataset = save['train_dataset'][1:FLAGS.max_steps]
      train_labels = save['train_labels'][1:FLAGS.max_steps]
      print ('Full training set', save['train_dataset'].shape)
      del save
  print(train_dataset.shape,'train_dataset.shape')
  print(train_labels.shape,'train_labels.shape')
  print(valid_dataset.shape,'valid_dataset.shape')
  print(valid_labels.shape,'valid_labels.shape')
  print(test_dataset.shape,'test_dataset.shape')
  print(test_labels.shape,'test_labels.shape')
            
  with tf.Graph().as_default():
      sess = tf.InteractiveSession()
      # Input placehoolders
      with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, FLAGS.image_size * FLAGS.image_size], name='x-input')
        image_shaped_input = tf.reshape(x, [-1, FLAGS.image_size, FLAGS.image_size, 1])
        tf.image_summary('input', image_shaped_input, 10)
        y_ = tf.placeholder(tf.float32, [None, FLAGS.num_classes], name='y-input')
        keep_prob = tf.placeholder(tf.float32)
        tf.scalar_summary('dropout_keep_probability', keep_prob)

      # We can't initialize these variables to 0 - the network will get stuck.
      def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

      def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

      def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.scalar_summary('mean/' + name, mean)
          with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
          tf.scalar_summary('sttdev/' + name, stddev)
          tf.scalar_summary('max/' + name, tf.reduce_max(var))
          tf.scalar_summary('min/' + name, tf.reduce_min(var))
          tf.histogram_summary(name, var)
     
      def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
          # This Variable will hold the state of the weights for the layer
          with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
          with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
          with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
          activations = act(preactivate, 'activation')
          tf.histogram_summary(layer_name + '/activations', activations)
          return activations

      # Train the model, and also write summaries.
      # Every 10th step, measure test-set accuracy, and write test summaries
      # All other steps, run train_step on training data, & add training summaries

      def feed_dict(dataset, labels, step, batch_size = 0, test = False):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if(batch_size == 0):
            batch_size = labels.size
        labels_size = int(labels.shape[0])
        batch_size = int(batch_size)
        if (labels_size - batch_size > 0):
            offset = (step* batch_size) % (labels_size - batch_size)
        else:
            offset = 0
        images = dataset[offset:(offset + batch_size), :]
        label_batch = labels[offset:(offset + batch_size)]
        # Map 2 to [0.0, 1.0, 0.0 ...], 3 to [0.0, 0.0, 1.0 ...]
        #indices = tf.reshape(tf.range(0, batch_size, 1), [-1, 1])
        #labels_hot = (np.arange(FLAGS.num_classes) == label_batch[:,None]).astype(np.float32)
        if(test == True):
            dropout = 1
        else:
            dropout = FLAGS.dropout
        return {x: images, y_: label_batch, keep_prob: dropout}
        
        
        
      hidden1 = nn_layer(x, FLAGS.image_size * FLAGS.image_size , 500, 'layer1')
      dropped = tf.nn.dropout(hidden1, keep_prob)
      y = nn_layer(dropped, 500, FLAGS.num_classes, 'layer2', act=tf.nn.softmax)

      with tf.name_scope('cross_entropy'):
        diff = y_ * tf.log(y)
        with tf.name_scope('total'):
          cross_entropy = -tf.reduce_mean(diff)
        tf.scalar_summary('cross entropy', cross_entropy)

      with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy)

      with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
          correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.scalar_summary('accuracy', accuracy)

      # Merge all the summaries 
      merged = tf.merge_all_summaries()
      train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train',
                                            sess.graph)
      test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test')
      tf.initialize_all_variables().run()

      #why this is here?
      for i in range(FLAGS.max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
          feed_dict2 = feed_dict(valid_dataset,valid_labels, i , 0 ,True)
          summaryTest = sess.run([merged], feed_dict=feed_dict2)
          #print(summaryTest,'summaryTest','aaa')
          #test_writer.add_summary(summaryTest, i)
        else:  # Record train set summarieis, and train
          #print('merged',merged)
          feed_dict1 = feed_dict(train_dataset,train_labels, i , 0 ,False)
          #print('feed_dict',feed_dict)
          summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict1)
          train_writer.add_summary(summary, i)
          print('Accuracy at step %s: %s' % (i, acc))


def main(_):
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  train()
'''
train_data, train_target, train_id = read_and_normalize_train_data(img_rows, img_cols)
test_data, test_id = read_and_normalize_test_data(img_rows, img_cols)
train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(
                                            train_data, train_target, test_size=0.4, random_state=42)
test_dataset, valid_dataset, test_labels, valid_labels = train_test_split(
                                            valid_dataset, valid_labels, test_size=0.5, random_state=42)
#convert to 1 hot encoding
#???

yfull_train = dict()
yfull_test = []

'''
if __name__ == '__main__':
  tf.app.run()

