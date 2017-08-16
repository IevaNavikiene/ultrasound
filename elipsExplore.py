import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

#from sklearn.linear_model import RANSACRegressor
#import pandas as pd

#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:

pickle_file = 'elips.pickle'
with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	elipseList = save['elipseList']
	filesList = save['filesList']
	del save

elipsCoordinates = np.zeros((len(elipseList),2))
elipsShape = np.zeros((len(elipseList),2))
angleList = np.zeros((len(elipseList),1))
thirdSideX = np.zeros((len(elipseList),1))
thirdSideY = np.zeros((len(elipseList),1))
i= 0
'''
xy - 0 : max - virsus desineje, min - gan kaireje , taigi cia pozicija x asyje
xy - 1 : max - apacia - istyrineti ar tokiu daug, min -virsus
centers -0 :max - apvalus stacias, min -plokscias - plotis max 115
centers -1 :max - plokscias, min -apvalus - aukstis max 280
angle: angle from 0 - 180
apacioje yra nedaug ir yra tarpas tarp apacios is vidurio
'''

for xy,centers,angle in elipseList:
	#print(x,'x',y,'y',z,'z')
	#elipsCoordinates = np.c_[elipsCoordinates,(xy[0],xy[1])]
	elipsCoordinates[i, :] = (xy[0],xy[1])
	elipsShape[i, :] = (centers[0],centers[1])
	if(angle > 90):
		angle = 180 - angle
	angleList[i] = angle

	thirdSideX[i] = np.sin(np.deg2rad(90 -angle)) * centers[0]/2
	thirdSideY[i] = np.sin(np.deg2rad(angle)) * centers[0]/2
	thirdSideY[i] = thirdSideY[i] *2
	thirdSideX[i] = thirdSideX[i] *2
		
	if(centers[1] >= 270):# labai apacioje essantis daiktas istryniau
		print(filesList[i], 'thirdSideX max file')
		print(centers, 'xy' , angle,'angle')
	'''	
	if(xy[0] <= 170):#20_26_mask.tif labai desineje virsuje esantis daiktas
		print(filesList[i], 'xy[0] max file')
	if(xy[1] >= 275):# labai apacioje essantis daiktas istryniau
		print(filesList[i], 'xy[1] max file')
	
	if(centers[0] >= 115): # ganetinai apvalus
		print(filesList[i], 'centers[0] max file')
	
	if(centers[1] >= 280): # labaai desineje horizontalus, istryniau (esme, kad pailgas)
		print(filesList[i], 'centers[1] max file')
	
	if(centers[0] <= 37): # ganetinai apvalus
		print(filesList[i], 'centers[0] min file')
	if(centers[1] <= 76): # labaai desineje horizontalus, istryniau (esme, kad pailgas)
		print(filesList[i], 'centers[1] min file')
	if(xy[0] <= 167):#20_26_mask.tif labai desineje virsuje esantis daiktas
		print(filesList[i], 'xy[0] min file')
	if(xy[1] <= 63):# labai apacioje essantis daiktas istryniau
		print(filesList[i], 'xy[1] min file')
	
	if(angle >= 100 and angle <= 120):
		print(filesList[i], 'angle[0] max file')
	if(angle <= 0.05):
		print(filesList[i], 'angle[0] min file')
	'''
	i = i+1
#print(elipsCoordinates,'elipsCoordinates')
'''
print(max(elipsCoordinates[:,0]),'elipsCoordinates[:,0]',max(elipsCoordinates[:,1]))
print(min(elipsCoordinates[:,0]),' min elipsCoordinates[:,0]',min (elipsCoordinates[:,1]))
print(max(angleList[:]),min(angleList[:]))
print(min(elipsShape[:,0]),' min elipsShape[:,0]',min(elipsShape[:,1]))
print(max(elipsShape[:,0]),'elipsShape[:,0]',max(elipsShape[:,1]))
'''

print(max(thirdSideX),'thirdSideX',max(thirdSideY))
val = 0
plt.plot(thirdSideY, np.zeros_like(thirdSideY) + val, 'x')
plt.show()
'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(elipsShape[:,0],elipsShape[:,1], angleList, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(elipsCoordinates[:,0],elipsCoordinates[:,1], angleList, c='b', marker='o')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.scatter(elipsCoordinates[:,0],elipsCoordinates[:,1], elipsShape[:,0], c='r', marker='o')
ax3.set_xlabel('X Label')
ax3.set_ylabel('Y Label')
ax3.set_zlabel('Z Label')
plt.show()


fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.scatter(elipsCoordinates[:,0],elipsCoordinates[:,1], elipsShape[:,1], c='b', marker='o')
ax4.set_xlabel('X Label')
ax4.set_ylabel('Y Label')
ax4.set_zlabel('Z Label')
plt.show()

fig5 = plt.figure()
ax5 = fig5.add_subplot(111, projection='3d')
ax5.scatter(elipsShape[:,0],elipsShape[:,1], elipsCoordinates[:,0], c='r', marker='o')
ax5.set_xlabel('X Label')
ax5.set_ylabel('Y Label')
ax5.set_zlabel('Z Label')
plt.show()


fig6 = plt.figure()
ax6 = fig6.add_subplot(111, projection='3d')
ax6.scatter(elipsShape[:,0],elipsShape[:,1], elipsCoordinates[:,1], c='b', marker='o')
ax6.set_xlabel('X Label')
ax6.set_ylabel('Y Label')
ax6.set_zlabel('Z Label')
plt.show()
'''
'''
n = 100
plt.scatter(elipsCoordinates[:,0],elipsCoordinates[:,1])
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
'''
# processing
'''
robust_estimator = RANSACRegressor(random_state=0)
robust_estimator.fit(np.vstack([a,b]).T, d)
d_pred = robust_estimator.predict(np.vstack([a,b]).T)

# calculate mse
mse = (d - d_pred.ravel()) ** 2

# get 50 largest mse, 50 is just an arbitrary choice and it doesn't assume that we already know there are 100 outliers
index = argsort(mse)
fig, axes = plt.subplots(ncols=2, sharey=True)
axes[0].scatter(a[index[:-50]], d[index[:-50]], c='b', label='inliers')
axes[0].scatter(a[index[-50:]], d[index[-50:]], c='r', label='outliers')

'''
'''
X = np.random.randint(25,50,(25,2))
Y = np.random.randint(60,85,(25,2))
Z = np.vstack((X,Y))

# convert to np.float32
Z = np.float32(Z)
 
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(Z,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now separate the data, Note the flatten()
A = Z[label.ravel()==0]
B = Z[label.ravel()==1]

# Plot the data
plt.scatter(A[:,0],A[:,1])
plt.scatter(B[:,0],B[:,1],c = 'r')
plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
plt.xlabel('Height'),plt.ylabel('Weight')
plt.show()
'''
