class Config():
	def __init__(self):
		self.pickle_file = 'elips.pickle'
		self.trainNervesPickle = 'masks9.pickle'
		self.elipseList = []
		self.filesList = []
		self.maxHeight = 70
		self.maxWidth = 60
		self.fullImageHeight = 420
		self.fullImageWidth = 580
		self.imageHeight = 9
		self.imageWidth = 14
		self.maskDir = './input/nervesMasks/'
		self.imgDir = './input/nerves/'
		self.imgFalseDir = './input/notNerves/'
		self.imgClassDir = './input/NervesClass/'
		self.imgFalseClassDir = './input/notNervesClass/'
		self.trainDir = './input/trainChanged/'
		self.modelWeightsFilePath = 'nerve_weights_tensorflow5.ckpt'
		self.masksPickle = 'masks3.pickle'
		self.defaultCentroidX = 200
		self.defaultCentroidY = 100
		self.partForValidation = 0.25
		self.partForTest = 0.25
		self.numClasses = 2
		self.batch_size = 128
		self.nbEpoch = 50
		self.randomState = 51
