# author: Adrian Rosebrock
# date: 27 January 2014
# website: http://www.pyimagesearch.com

# USAGE
# python search.py --dataset images --index index.cpickle

# import the necessary packages
from loadimage.searcher import Searcher
import numpy as np
import argparse
import cPickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where we stored our index")
args = vars(ap.parse_args())
dataset = "./input/trainChanged2"
# load the index and initialize our searcher
index = cPickle.loads(open(args["index"]).read())
searcher = Searcher(index)

# loop over images in the index -- we will use each one as
# a query image
for (query, queryFeatures) in index.items():
	# perform the search using the current query
	results = searcher.search(queryFeatures)

	# load the query image and display it
	path = dataset + "/%s" % (query)
	queryImage = cv2.imread(path)
	#cv2.imshow("Query", queryImage)
	print "query: %s" % (query)

	# initialize the two montages to display our results --
	# we have a total of 25 images in the index, but let's only
	# display the top 10 results; 5 images per montage, with
	# images that are 400x166 pixels
	montageA = np.zeros((420 * 5, 580, 3), dtype = "uint8")
	montageB = np.zeros((420 * 5, 580, 3), dtype = "uint8")

	# loop over the top ten results
	for j in xrange(0, 10):
		# grab the result (we are using row-major order) and
		# load the result image
		(score, imageName) = results[j]
		path = dataset + "/%s" % (imageName)
		result = cv2.imread(path)
		print "\t%d. %s : %.3f" % (j + 1, imageName, score)

		# check to see if the first montage should be used
		if j < 5:
			montageA[j * 420:(j + 1) * 420, :] = result

		# otherwise, the second montage should be used
		else:
			montageB[(j - 5) * 420:((j - 5) + 1) * 420, :] = result

# show the results
cv2.imshow("Results 1-5", montageA)
cv2.imshow("Results 6-10", montageB)
cv2.waitKey(0)
