import skimage
import numpy as np
import cv2

def reshape_img(img,name):
    mon_image = skimage.io.imread(img)
    skimage.io.imsave('static/'+name, mon_image)
    mon_image = cv2.resize(mon_image, (28,28))
    mon_image = mon_image / 255.0
    mon_image = mon_image.reshape(-1,28,28,1)
    return mon_image
