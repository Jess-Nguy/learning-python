"""
The purpose of this is using the best inertia to quantize the colour for the second image.
Jess Nguyen
"""

import numpy as np
from skimage import data
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Initializing first image
plt.figure()
eye = io.imread("eye.jpg")
plt.title("eye.jpg")
plt.imshow(eye)
plt.axis('off')
plt.show()
inertiaArray = []
count = 0
kvalue = []

# Increments 2 KMean
for k in range(0, 9):
    flateye = eye.reshape((720*1280, 3))
    count += 2
    km = KMeans(n_clusters=count)
    km.fit(flateye)
    clusters = km.cluster_centers_
    inertia = km.inertia_
    prediction = km.predict(flateye)
    inertiaArray.append(inertia)
    if count == 10:
        kvalue = clusters

    flateye = eye.reshape((720, 1280, 3))
    newcolours = clusters[prediction].astype(int)
    newcolours = newcolours.reshape((720, 1280, 3))
    plt.figure()
    plt.title("Eye.jpg K="+str(count))
    plt.imshow(newcolours)
    plt.axis('off')
    plt.show()
    print("k=", count, "sse =", inertia)

# Initializing second image
plt.figure()
onepiece = io.imread("onepiece.jpg")
plt.title("onepiece.jpg")
plt.imshow(onepiece)
plt.axis('off')
plt.show()

# Inertia for second image
flatonepiece = onepiece.reshape((720*1280, 3))
km = KMeans(n_clusters=10)
km.fit(flatonepiece)
prediction = km.predict(flatonepiece)
newcoloursonepiece = kvalue[prediction].astype(int)
newcoloursonepiece = newcoloursonepiece.reshape((720, 1280, 3))
plt.figure()
plt.title("onepiece.jpg "+str(10))
plt.imshow(newcoloursonepiece)
plt.axis('off')
plt.show()

# Quantative Chart for 1st image
plt.figure()
tickLabels = [2, 4, 6, 8, 10, 12, 14, 16, 18]
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], labels=tickLabels)
plt.plot(inertiaArray)
plt.show()

# save K=10 best inertia for second image
io.imsave("onepieceinertia.jpg", newcoloursonepiece)
