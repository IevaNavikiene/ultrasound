import cv2
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import glob
from configFileTEST import Config
from NervesFileTEST import Nerves

import tensorflow as tf
configModel = Config()
dataModel = Nerves(configModel)

dataModel.prepareDataTolearn()
