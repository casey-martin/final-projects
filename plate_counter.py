from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.morphology import watershed, disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from sklearn.cluster import KMeans

# not fit for human consumption

# import tif and convert to RGB for pyplot usage
mytif = cv2.imread("/home/dalgarno/Documents/GAW/final-projects/1_3 50ul (g).tif", 1)
mytif = cv2.cvtColor(mytif, cv2.COLOR_BGR2RGB)

#convert tif to grayscale
graytif = cv2.cvtColor(mytif, cv2.COLOR_BGR2GRAY)
image = img_as_ubyte(graytif)

# denoise image
denoised = rank.median(image, disk(2))

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers = rank.gradient(denoised, disk(2)) < 10
markers = ndi.label(markers)[0]

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(2))

# process the watershed
labels = watershed(gradient, markers)

ret, thresh = cv2.threshold(labels.astype(np.uint8), 3, 255, cv2.THRESH_BINARY_INV)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


mycentroids = []
index = 0
for c in contours:
	# compute the center of the contour
    M = cv2.moments(c)
    if M["m00"] > 0:
        index +=1
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #if sum(mytif[cX, cY]) != 0:

        mycentroids.extend([cX, cY])

        cv2.drawContours(mytif, [c], -1, (0, 255, 0), 2)
        cv2.circle(mytif, (cX, cY), 7, (255, 255, 255), -1)




#centroid binnning. Work in progress. Perhaps filter by 
mycentroids = [(x,y) for x,y in zip(mycentroids[::2], mycentroids[1::2])]

rgb_list = []
for i in range(len(mycentroids)):
    centroid_rgb = mytif[mycentroids[i][0], mycentroids[i][1]]
    rgb_list.extend([centroid_rgb])

mykmeans = KMeans(n_clusters=2)
mykmeans.fit(rgb_list)

clusters = dict(zip(mycentroids, mykmeans.labels_))


indexX = 0
indexY = 0
for i in range(len(mycentroids)):
    if mykmeans.labels_[i] == 1:
        indexY += 1
        myindex = indexY
        txtclr = (255, 255, 255)


    else:
        indexX += 1
        myindex = indexX
        txtclr = (0, 0, 0)

    cv2.putText(mytif, str(myindex), (mycentroids[i][0] - 20, mycentroids[i][1] - 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, txtclr, 3)

	# show the image

myout = plt.imshow(mytif)
plt.show()

