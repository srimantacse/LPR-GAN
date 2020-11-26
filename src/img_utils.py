import os
import cv2
import sys
import numpy as np
from PIL import Image
import os.path as path
import matplotlib.pyplot as plt
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import imutils
from scipy import ndimage

from skimage import measure, morphology
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops

def image_clean(img):  
  kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
  kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
  kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
  kernel_4 = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))


  img_1    = img
  _, img_1 = cv2.threshold(img_1, 127, 255, cv2.THRESH_BINARY) 
  #img_1 = cv2.dilate(img_1, kernel_3) 
  #img_1 = cv2.erode (img_1, kernel_3)

  return img_1

def get_batch( X_curr, Y_curr, path_X, path_Y, shape=(128,128)):
  x_data = []
  y_data = []  
  
  for i in range(len(X_curr)):
    
    img_x = cv2.imread(os.path.join(path_X, X_curr[i]))
    img_y = cv2.imread(os.path.join(path_Y, Y_curr[i]), cv2.IMREAD_GRAYSCALE)

    img_x = cv2.resize(img_x, shape, interpolation = cv2.INTER_CUBIC)
    img_y = cv2.resize(img_y, shape, interpolation = cv2.INTER_CUBIC)

    # Binarized Train Output
    img_y = image_clean(img_y)

    img_x = np.array(img_x).astype('float32')/255
    img_y = np.array(img_y).astype('float32')/255

    x_data.append(img_x)
    y_data.append(img_y)
  
  x_data = np.array(x_data)
  y_data = np.array(y_data)

  return x_data, y_data

def print_images(img_list, img_label_list):
  noOfImages = len(img_list)
  plt.figure(figsize=(noOfImages*5,15))
  imageCount = 1

  for img in img_list:
    plt.subplot(1,noOfImages,imageCount)
    plt.subplot(1,noOfImages,imageCount).set_title(img_label_list[imageCount-1])
    plt.imshow(img, cmap='gray')
    imageCount += 1 

  plt.show()

def fetchMask(gray_image):
  dilateList = []
  mask   = np.ones(gray_image.shape, dtype=np.uint8) * 255
  thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  dilate = cv2.dilate(thresh, kernel, iterations=3)

  cnts   = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts   = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      area = cv2.contourArea(c)
      if area < 5000:
          x,y,w,h = cv2.boundingRect(c)
          mask[y:y+h, x:x+w] = gray_image[y:y+h, x:x+w]

  dilate = dilate /255
  dilate_1 = dilate.copy()
  dilate_1[np.nonzero(dilate.sum(axis=1) < 50), :] = 0

  dilateList.append(dilate)
  dilateList.append(1 - (dilate - dilate_1))

  return dilateList

def clean_image_1(gray_image, thresh_value=127):
  # Input expected [0, 255] thresholded image
  mask       = fetchMask(gray_image)

  maskedImg  = np.multiply(gray_image, mask[1])
  maskedImg  = 255 - np.abs(maskedImg - mask[1]*255)

  _, maskedImg = cv2.threshold(maskedImg, thresh_value, 255, cv2.THRESH_BINARY)

  return mask[1], maskedImg.astype('uint8')


def drd_fn(im, im_gt):
	height, width = im.shape
	neg = np.zeros(im.shape)
	neg[im_gt!=im] = 1
	y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
	
	n = 2
	m = n*2+1
	W = np.zeros((m,m), dtype=np.uint8)
	W[n,n] = 1.
	W = cv2.distanceTransform(1-W, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	W[n,n] = 1.
	W = 1./W
	W[n,n] = 0.
	W /= W.sum()
	
	nubn = 0.
	block_size = 8
	for y1 in range(0, height, block_size):
		for x1 in range(0, width, block_size):
			y2 = min(y1+block_size-1,height-1)
			x2 = min(x1+block_size-1,width-1)
			block_dim = (x2-x1+1)*(y1-y1+1)
			block = 1-im_gt[y1:y2, x1:x2]
			block_sum = np.sum(block)
			if block_sum>0 and block_sum<block_dim:
				nubn += 1

	drd_sum= 0.
	tmp = np.zeros(W.shape)
	for i in range(min(1,len(y))):
		tmp[:,:] = 0 

		x1 = max(0, x[i]-n)
		y1 = max(0, y[i]-n)
		x2 = min(width-1, x[i]+n)
		y2 = min(height-1, y[i]+n)

		yy1 = y1-y[i]+n
		yy2 = y2-y[i]+n
		xx1 = x1-x[i]+n
		xx2 = x2-x[i]+n

		tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
		tmp *= W

		drd_sum += np.sum(tmp)
	return drd_sum/nubn


def clean_image_2(thresh, thresh_value=127):
  #_, thresh = cv2.threshold(thresh, thresh_value, 255, cv2.THRESH_BINARY)

  labels    = measure.label(thresh, connectivity=2, background=255)
  mask      = np.zeros(thresh.shape, dtype="uint8")
    
  for label in np.unique(labels):
    if label == 255:
      continue

    # otherwise, construct the label mask to display only connected components for the
    # current label, then find contours in the label mask
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts  = cnts[0] if len(cnts) == 2 else cnts[1]
    #print(cnts)

    if len(cnts) > 0:
      # grab the largest contour which corresponds to the component in the mask, then
      # grab the bounding box for the contour
      c = max(cnts, key=cv2.contourArea)
      (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

      # compute the aspect ratio, solidity, and height ratio for the component
      aspectRatio = boxW / float(boxH)
      solidity = cv2.contourArea(c) / float(boxW * boxH)
      heightRatio = boxH / float(thresh.shape[0])

      # determine if the aspect ratio, solidity, and height of the contour pass
      # the rules tests
      keepAspectRatio = aspectRatio < 1.0
      keepSolidity    = solidity > 0.05
      keepHeight      = heightRatio > 0.20 and heightRatio < 0.80

      if keepAspectRatio and keepSolidity and keepHeight:
        hull = cv2.convexHull(c)
        cv2.drawContours(mask, [hull], -1, 255, -1)

  mask = segmentation.clear_border(mask)
  mask = (mask/255).astype('uint8')
  genImage = 255 - np.zeros(thresh.shape, dtype="uint8")
  genImage[mask==1] = thresh[mask==1]

  #_, genImage = cv2.threshold(genImage, thresh_value, 255, cv2.THRESH_BINARY)

  return mask, genImage

def clean_image_3(img):
  img_copy = img
  org_shape = img.shape
  img  = cv2.resize(img, (512,512))
  _, blackAndWhite = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)

  nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
  sizes = stats[1:, -1]
  img2 = np.zeros((labels.shape), np.uint8)
  for i in range(0, nlabels - 1):
      if sizes[i] >= np.average(sizes) * 256:   #filter small dotted regions
          img2[labels == i + 1] = 255

  sumArray_col = np.sum(img2/255, axis=0)
  colwise_index = [n for n, i in enumerate(sumArray_col) if i > img2.shape[1]/2]
  
  sumArray_row = np.sum(img2/255, axis=1)
  rowwise_index = [n for n, i in enumerate(sumArray_row) if i > img2.shape[0]/2]
  
  mask = np.zeros((labels.shape), np.uint8)
  mask[:,colwise_index] = 255
  mask[rowwise_index,:] = 255
  
  mask  = cv2.resize(mask, (org_shape[1], org_shape[0]))    
  ret, result = cv2.threshold(img_copy, 127, 255, cv2.THRESH_BINARY)
  result[mask==255] = 255
  
  return mask, result


def grayscale_dialation(img, thresh=245):
  gray_dial = ndimage.grey_dilation(img, size=(5,5), structure=np.ones((5, 5)))
  _, gray_dial = cv2.threshold(gray_dial,thresh,255,cv2.THRESH_BINARY)
  return gray_dial

def connected_component(img, thresh=127, eosionDialReq=True):
    img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    
    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):
        #print(region.area)
        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        # print region.area # (for debugging)
        # take regions with large enough areas
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)
    #print("the_biggest_component: " + str(the_biggest_component))
    #print("average: " + str(average))

    #a4_constant = ((average/84.0)*250.0)+100
    a4_constant = average/10 + 100

    #print("a4_constant: " + str(a4_constant))
    b = morphology.remove_small_objects(blobs_labels, a4_constant)


    #plt.imsave('pre_version.png', b)
    img = b
    #img = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    #print(img[0][0])

    img_1 = np.zeros(img.shape)
    img_1[img == img[0][0]] = 255

    if eosionDialReq:
        kernel = 2
        print("Dilation Done...", kernel)
        img_1 = cv2.dilate(img_1/255, np.ones((kernel,kernel)))
        #img_1 = cv2.erode(img_1/255, np.ones((kernel,kernel)))
        img_1 = img_1 * 255

    return img_1
