# coding: utf-8
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import cv2
import numpy as np
filename = 'submissionNr1.csv'
test_data_dir = './input/test/'
totalShow = 4

def showResults(filename, totalShow):
	print('Preparing arrays...')
	f = open(filename, "r")
	f.readline()
	img = defaultdict(lambda: defaultdict(int))
	pixels = defaultdict(lambda: defaultdict(int))
	total = 0
	# Calc counts
	while 1:
		line = f.readline().strip()

		if total % 1000000 == 0:
			print('Read {} lines...'.format(total))

		if line == '':
			break

		arr = line.split(",")
		imgNum = int(arr[0])
		img = findImage(imgNum)
		pixels = arr[1]
		mask = pixelsToImage(pixels, img.shape)
		#cv2.imshow('moskito',mask)
		#cv2.waitKey(0)
		maskNotBlank = mask_not_blank(mask)
		if(maskNotBlank == True):
			showImageWithContour(img, mask)
			total += 1
		if(total > totalShow):
			break
	f.close()

	print('Good luck!')

def image_with_mask(img, mask):
	mask_edges = cv2.Canny(mask, 1, 255)>0
	img[mask_edges, 0] = 0
	img[mask_edges, 1] = 255
	img[mask_edges, 2] = 0

	return img

def fimg_to_fmask(img_path):
	# convert an image file path into a corresponding mask file path 
	dirname, basename = os.path.split(img_path)
	maskname = basename.replace(".tif", "_mask.tif")
	return os.path.join(dirname, maskname)

def mask_not_blank(mask):
	return sum(mask.flatten()) > 0
   
def findImage(imgNumber):
	path = test_data_dir + str(imgNumber) + ".tif"
	img = cv2.imread(path)
	return img
	
def grays_to_RGB(img):
	# turn 2D grayscale image into grayscale RGB
	return np.dstack((img, img, img)) 
  
def showImageWithContour(img, mask):
	colorImage = image_with_mask(img, mask)
	cv2.imshow('colorImage',colorImage)
	cv2.waitKey(0)
 
def pixelsToImage(row, defaultImageSize):
	image = np.zeros(defaultImageSize[0] * defaultImageSize[1], dtype=np.uint8)
	encodedImage = row.split(' ') 
	imageDecoded = []
	# loop through every image row
	rowDecoded = ''
	count = 0
	char = encodedImage[0]
	# loop through every character in the row
	# and count how many consecutive elements of 
	# that character are on that image row encodedImage
	#print('encodedImage',defaultImageSize[0] ,defaultImageSize[1],defaultImageSize[0] * defaultImageSize[1])
	i = 0
	lastCharacter = 0
	for character in encodedImage:
		i += 1
		# switching to a new character
		if(character != ' ' and character != ''):
			if(i % 2 == 0 and character !=0):
				image[lastCharacter - 1:lastCharacter + int(character) - 1] = 255
				#runs.append((lastCharacter, int(character)))
			else:
				lastCharacter = int(character)
	image = image.reshape((defaultImageSize[0],defaultImageSize[1]) , order='F')
	#print(encodedImage,'encodedImage')
	return image

showResults(filename, totalShow)
