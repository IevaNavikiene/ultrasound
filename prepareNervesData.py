import cv2
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import os
import glob
# imu x kordinate pridedu ir atimu po puse plocio...
# iskerpu si kvadrata pagal pixelius is nuotraukos ir is kaukes ir issaugau i atitinkamas papkes

class prepareNervesData:
	def __init__(self):
		self.pickle_file = 'elips.pickle'
		self.elipseList = []
		self.filesList = []
		self.maxHeight = 140
		self.maxWidth = 120
		self.maskDir = './input/nervesMasks/'
		self.imgDir = './input/nerves/'
		self.imgFalseDir = './input/notNerves/'
		self.trainDir = './input/trainChanged/'
		self.defaultCentroidX = 200
		self.defaultCentroidY = 100
		
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
							
class inference_hidden0(Nerves):
	def __init__(self):
		Nerves.__init__(self)
############main method#########################
if __name__ == '__main__':
	model0 = inference_hidden0()
	#model0.extractNerves()
	model0.extractNotNerves()
