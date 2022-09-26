# In this tutorial, we will use a matrix decomposition from linear algebra, the Singular Value Decomposition, 
# to generate a compressed approximation of an image. We’ll use the face image from the scipy.misc module:
# img.shape: returns the dimension of the matrix, where the dimension is the number of elements. The image is RGB, hence is a 3 dimensions matrix. 

from scipy import misc
import numpy
import matplotlib.pyplot as plt

img = misc.face()

print(type(img))

#numpy.ndarray
plt.imshow(img)
plt.show()

print(img.shape)
print(img.ndim)