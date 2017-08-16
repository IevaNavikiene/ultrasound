import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
from six.moves import cPickle as pickle

# Definitions
IMAGES_TO_SHOW = 1

# Plot image
def plot_image(img1, img2, title=None):
	showImage = cv2.add(img1,img2)
	plt.figure(figsize=(15,20))
	plt.title(title)
	plt.imshow(showImage)
	plt.show()

# Draw elipsis on image
def draw_ellipse(mask):
	ret, thresh = cv2.threshold(mask, 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, 1, 2)
	m3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
	has_ellipse = len(contours) > 0
	if has_ellipse:
		cnt = contours[0]
		ellipse = cv2.fitEllipse(cnt)
		#print(ellipse,'ellipse')
		cx, cy = np.array(ellipse[0], dtype=np.int)# suklasifikuoti visus cx ir cy
		m3[cy-2:cy+2,cx-2:cx+2] = (255, 0, 0)
		cv2.ellipse(m3, ellipse, (0, 255, 0), 1)
		
	return has_ellipse, m3, ellipse
	
# Read some files
mfiles = glob.glob("./input/trainChanged/*_mask.tif")
#random.shuffle(mfiles) # Shuffle for random results

files_with_ellipse = 0
elipseList = []
filesList = []
for mfile in mfiles:
	mask = cv2.imread(mfile, -1)  # imread(..., -1) returns grayscale images
	if (mask.sum() != 0):
		has_ellipse, mask_with_ellipse, ellipse = draw_ellipse(mask)
		elipseList.append(ellipse)
		filesList.append(mfile)
		'''
		if (has_ellipse):
			files_with_ellipse = files_with_ellipse+1
			flbase = os.path.basename(mfile)
			imageFile = "./input/trainChanged/" + flbase[:-9] + ".tif"
			image = cv2.imread(imageFile)  # imread(..., -1) returns grayscale images
			plot_image(mask_with_ellipse, image, 'noName')
			if files_with_ellipse > IMAGES_TO_SHOW:
				break
		'''
try:
  f = open('elips.pickle', 'wb')
  save = {
	'elipseList': elipseList,
	'filesList' : filesList,
	}
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to elis:', e)
  raise
