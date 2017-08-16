# import the necessary packages
import imutils
#python SIFT algorithm - paskaityti reikia 
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image
		
def pyramid2(image, scale=1.5, minSize=(30, 30)):
	# METHOD #2: Resizing + Gaussian smoothing.
	for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
		# if the image is too small, break from the loop
		if resized.shape[0] < 30 or resized.shape[1] < 30:
			break
			
		# show the resized image
		cv2.imshow("Layer {}".format(i + 1), resized)
		cv2.waitKey(0)

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			#print(x + windowSize[1],'windowSize[1]',image.shape[1],'image.shape[1]')
			#print(windowSize[1],'windowSize[1]',x windowSize[0],'image.shape[1]')
			if(x + windowSize[1] > image.shape[1]):
				yield (x, y, image[image.shape[1] - windowSize[1]:image.shape[1], x:x + windowSize[0]])
			else:
				yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# import the necessary packages
'''
from pyimagesearch.helpers import pyramid

from skimage.transform import pyramid_gaussian
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())
 
# load the image
image = cv2.imread(args["image"])
cv2.imshow("image",image)
cv2.waitKey(0)
# METHOD #1: No smooth, just scaling.
# loop over the image pyramid
for (i, resized) in enumerate(pyramid(image, scale=args["scale"])):
	# show the resized image
	cv2.imshow("Layer {}".format(i + 1), resized)
	cv2.waitKey(0)
 
# close all windows
cv2.destroyAllWindows()
 
# METHOD #2: Resizing + Gaussian smoothing.
for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
	# if the image is too small, break from the loop
	if resized.shape[0] < 30 or resized.shape[1] < 30:
		break
		
	# show the resized image
	cv2.imshow("Layer {}".format(i + 1), resized)
	cv2.waitKey(0)
'''



# import the necessary packages
#from pyimagesearch.helpers import pyramid
#from pyimagesearch.helpers import sliding_window
'''
import argparse
import time
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (128, 128)

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
 
		# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
		# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
		# WINDOW
 
		# since we do not have a classifier, we'll just draw the window
		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.025)
'''
