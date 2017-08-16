from loadimage.histogram import Histogram
import argparse
import cPickle
import glob
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required = True,
#    help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
    help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())
print('args',args)
# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = Histogram([8])

# use glob to grab the image paths and loop over them
allImages = glob.glob("./input/trainChanged2/*[0-9].tif")
print('allImages',allImages)
for imagePath in allImages:

    # extract our unique image ID (i.e. the filename)
    k = imagePath[imagePath.rfind("/") + 1:]
    image = cv2.imread(imagePath)
    flbase = os.path.basename(imagePath)
    mask_path = "./input/trainChanged2/" + flbase[:-4] + "_mask.tif"
    mask = cv2.imread(mask_path)
    features = desc.describe(image, mask)
    index[k] = features
# we are now done indexing our image -- now we can write our
# index to disk
f = open(args["index"], "w")
f.write(cPickle.dumps(index))
f.close()

# show how many images we indexed
print "done...indexed %d images" % (len(index))
