# Python program to illustrate foreground extraction using GrabCut algorithm

# organize imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
 
# path to input image specified and 
# image is loaded with imread command
image = cv2.imread('data/leon.jpg')

height, width = image.shape[:2]
print(f'Dimensiones de la imagen: {width}x{height}')
 
# create a simple mask image similar
# to the loaded image, with the 
# shape and return type
mask = np.zeros(image.shape[:2], np.uint8)
 
# specify the background and foreground model
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)
 
# define the Region of Interest (ROI) de forma proporcional al
# tamaño real de la imagen
margin = int(min(width, height) * 0.05)
rectangle = (margin, margin, width - margin * 2, height - margin * 2)
 
# apply the grabcut algorithm
cv2.grabCut(image, mask, rectangle,  
            backgroundModel, foregroundModel,
            5, cv2.GC_INIT_WITH_RECT)
 
# Mask final binaria
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
 
# The final mask is multiplied with 
# the input image to give the segmented image.
image_segmented = image * mask2[:, :, np.newaxis]

# Guardar imagen segmentada en carpeta out
os.makedirs('out', exist_ok=True)
cv2.imwrite('out/leon_segmented.jpg', image_segmented)
print('Imagen guardada en: out/leon_segmented.jpg')
 
# output segmented image with colorbar
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(cv2.cvtColor(image_segmented, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()