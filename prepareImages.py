import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
np.random.seed(212)
import cv2
import os
import glob
import datetime
import time
from sklearn.cross_validation import KFold
from keras.utils import np_utils
from six.moves import cPickle as pickle

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_string('training_data_dir', './input/trainChanged2/', 'Directory for storing data')
flags.DEFINE_string('test_data_dir', './input/test2/', 'Directory for storing data')
flags.DEFINE_string('image_size', 32, '')
flags.DEFINE_string('pickle_file', 'ultrasound.pickle', 'Save data in this file')
flags.DEFINE_string('pickle_file_tests', 'ultrasoundTEST.pickle', 'Save data in this file')

#First try on resized, then try on full images
def get_im_cv2(path, img_rows, img_cols):
    img = cv2.imread(path,0)
    resized = cv2.resize(img, (img_cols, img_rows), interpolation = cv2.INTER_LINEAR)
    return resized

def load_image(folder, test):
  """Load the data for a single letter label."""
  image_files = glob.glob(folder + "*[0-9].tif")
  dataset = np.ndarray(shape=(len(image_files), FLAGS.image_size, FLAGS.image_size),
                         dtype=np.float32)
  mask_train = []
  print(folder)
  num_images = 0
  for image_file in image_files:
    #image_file = os.path.join(folder, image)
    try:
      dataset[num_images, :, :] = get_im_cv2(image_file, FLAGS.image_size, FLAGS.image_size)
      if(test == False):
		  flbase = os.path.basename(image_file)
		  mask_path = "./input/trainChanged/" + flbase[:-4] + "_mask.tif"
		  mask = get_im_cv2(mask_path, FLAGS.image_size, FLAGS.image_size)
		  mask_train.append(mask)
		  
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]   
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset, mask_train
        
def maybe_pickle(folder,set_filename, test=False):
  dataset_names = []
  dataset_names.append(set_filename)
  labels = []
  if os.path.exists(set_filename):
     # You may override by setting force=True.
     print('%s already present - Skipping pickling.' % set_filename)
     return []
  else:
     print('Pickling %s.' % set_filename)
     dataset, labels = load_image(folder, test)
     reshapedDataset = dataset.reshape(dataset.shape[0],dataset.shape[1]*dataset.shape[2])
     print(reshapedDataset.shape,'new shape')
     #print(labels,'labelslabelslabels')
     try:
       with open(set_filename, 'wb') as f:
         #print(dataset[0],'daaaa')
         #print(labels[0],'labelszzzs',set_filename,'set_filename')
         pickle.dump({'dataset':reshapedDataset,'labels':labels}, f, pickle.HIGHEST_PROTOCOL)
     except Exception as e:
       print('Unable to save data to', set_filename, ':', e)
     return dataset, labels
   
  
# returns array of mask types (zero if no nerve is detected)
# @param array mask
# @return array 
def get_empty_mask_state(mask):
    out = []
    #print('mask-get_empty_mask_state',mask)
    for i in range(len(mask)):
        if mask[i].sum() == 0:
            out.append(0)
        else:
            out.append(1)
    return np.array(out)

train_dataset,train_labels = maybe_pickle(FLAGS.training_data_dir,FLAGS.pickle_file)
train_labels = get_empty_mask_state(train_labels)
train_labels = np_utils.to_categorical(train_labels, 2)

test_datasets_no_labels = maybe_pickle(FLAGS.test_data_dir,FLAGS.pickle_file_tests, test=True)

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
  
train_dataset, train_labels = randomize(train_dataset, train_labels)
train_dataset = train_dataset.reshape(train_dataset.shape[0],train_dataset.shape[1]*train_dataset.shape[2])
print(train_datasets.shape,'new shape')
#divide dataset to cross-validation and test set
train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(
                                           train_dataset, train_labels, test_size=0.4, random_state=42)
test_dataset, valid_dataset, test_labels, valid_labels = train_test_split(
                                            valid_dataset, valid_labels, test_size=0.5, random_state=42)
try:
  f = open(FLAGS.pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    'test_datasets_no_labels': test_datasets_no_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

