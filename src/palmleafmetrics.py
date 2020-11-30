# Commented out IPython magic to ensure Python compatibility.
import os
import cv2
import math
import numpy as np
from keras import models
from google.colab import drive
from keras.models import Model
from sklearn.metrics import precision_score, f1_score, multilabel_confusion_matrix
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import matplotlib.pyplot as plt
# %matplotlib inline

def get_patches(img, pref_height=240, pref_width=480, gray=False):
  c = 1
  if gray:
    h_org, w_org = img.shape
  else:
    h_org, w_org, c_org = img.shape
    c = c_org
  h, w = h_org + (pref_height - h_org % pref_height), w_org + (pref_width  - w_org % pref_width)

  if gray:
    result = np.full((h,w),   255, dtype=np.uint8)
  else:
    result = np.full((h,w,c), 255, dtype=np.uint8)

  result[0:h_org, 0:w_org] = img

  patches = []
  i       = 0
  while (i+pref_height <= h):
    temp = []
    j    = 0
    while (j+pref_width <= w):
      patch = result[i:i+pref_height, j:j+pref_width]
      temp.append(patch)
      j += pref_width
    patches.append(temp)
    i += pref_height	

  return patches, h, w, c

def concat_patches( patches, h, w, c, pref_height=240, pref_width=480):
	result = np.full((h,w), 255, dtype=np.uint8)
	row    = 0
	col    = 0
	i      = 0

	while i < len(patches):
		j   = 0
		col = 0
		while j < len(patches[0]):
			result[row:row+pref_height, col:col+pref_width] = patches[i][j] 
			col += pref_width
			j   += 1
		row += pref_height
		i   += 1

	return result

def crop_to_original_size(img_org, img_out):
  h, w, c = img_org.shape
  img_out = img_out[0:h, 0:w]

  return img_out

def plot_patches(patches):
	fig=plt.figure(figsize=(10,5))
	columns = len(patches[0])
	rows    = len(patches)
 
	for i in range(0, rows):
		for j in range(0, columns):
			pos = i * columns + j + 1
			img = patches[i][j]
			fig.add_subplot(rows, columns, pos)
			plt.imshow(img, cmap='gray')
	plt.show()
 
def plot_image(img):
	figs = plt.figure(figsize=(10,5))
	figs.add_subplot(1, 1,1)
	plt.imshow(img, cmap='gray')
	plt.show()

def predict_output(patches):
	output_patches = []
	for list1 in patches:
		temp = []	
		for img in list1:
			img = np.array([img]).astype('float32')/255
			img_output = model.predict(img)
			img_output *= 255
			img_output = img_output.astype('uint8')
			temp.append(img_output[0][:,:,0])

		output_patches.append(temp)
	return output_patches

# Palm Leaf With Output creation
test_dir        = '/content/gdrive/MyDrive/Code/PalmLeaf/Test/'
output_test_dir = '/content/gdrive/MyDrive/Code/PalmLeaf/Output_GT2/'
model_path      = '/content/gdrive/MyDrive/Code/palm_leaf_gen_latest.h5'

model = models.load_model(model_path)

output_patches = []
result         = []
def predict(test_dir, output_test_dir):
	for filename in sorted( os.listdir(test_dir)):	
		img              = cv2.imread( os.path.join(test_dir, filename))
		patches, h, w, c = get_patches(img)
		output_patches   = predict_output(patches)
		result           = concat_patches(output_patches, h, w, c)
		img_out          = crop_to_original_size(img, result)

		#cv2.imwrite( os.path.join(output_test_dir, filename[:-3]+"bmp"), img_out)  
		#print(filename[:-3]+"bmp")
	
	return output_patches, img_out

output_patches, result = predict(test_dir, output_test_dir)

def my_psnr(output_patches, gt_patches):
  columns = len(output_patches[0])
  rows    = len(output_patches)

  psn    = 0
  fsc    = 0
  nrmVal = 0
  count  = 0
  count1 = 0


  for i in range(0, rows):
    for j in range(0, columns):
      _, x1 = cv2.threshold(output_patches[i][j], 127, 255, cv2.THRESH_BINARY)
      _, x2 = cv2.threshold(gt_patches[i][j],     127, 255, cv2.THRESH_BINARY)

      val  = compare_psnr(output_patches[i][j], gt_patches[i][j], data_range=255)
      fsc += f1_score(x1.flatten()/255, x2.flatten()/255, average='weighted')
      if math.isinf(val) == False:
        psn += val
        count +=1

      val1 = nrm(x1.flatten(), x2.flatten())
      if val1 != 0:
        nrmVal += val1
        count1 +=1
      else:
        c11 = 0

  return (psn/ count), (fsc/ count), (nrmVal/ count1)

def nrm(y_true, y_pred):
  mcm = multilabel_confusion_matrix(y_true, y_pred)
  tn = mcm[:, 0, 0][0]
  tp = mcm[:, 1, 1][0]
  fn = mcm[:, 1, 0][0]
  fp = mcm[:, 0, 1][0]

  if fn +  tp == 0:
    nr_fn = 0.0
  else:
    nr_fn   = fn.astype(float) / (fn +  tp)

  if fp  + tn == 0:
    nr_fp = 0.0
  else:
    nr_fp = fp.astype(float) / (fp +  tn)

  nrm_val = (nr_fn + nr_fp) / 2.0

  return nrm_val

# Overall Test
test_dir        = '/content/gdrive/MyDrive/Code/PalmLeaf/Test/'
gt_dir          = '/content/gdrive/MyDrive/Code/PalmLeaf/GT/'
output_test_dir = '/content/gdrive/MyDrive/Code/PalmLeaf/Output/'
model_path      = '/content/gdrive/MyDrive/Code/palm_leaf_gen_latest.h5'

model = models.load_model(model_path)

psnr_list = []
fsc_list  = []
nrm_list  = []
for filename in sorted( os.listdir(output_test_dir)):
  # Output Image
  out_img = cv2.imread(os.path.join(output_test_dir, filename), 0)
  output_patches1, h, w, c = get_patches(out_img, 45, 45, True)

  # GT Image 1
  gt_image_path = os.path.join(gt_dir, filename[:-4]+"_GT1.bmp")
  gt_img        = cv2.imread(gt_image_path,     cv2.IMREAD_GRAYSCALE)
  _, gt_img     = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)
  gt_patches1, h, w, c = get_patches(gt_img, 45, 45, True)

  _psn, _fsc, _nrm = my_psnr(output_patches1, gt_patches1)

  psnr_list.append(_psn)
  fsc_list.append(_fsc)
  nrm_list.append(_nrm)
  print(filename, '\t', _psn, _fsc, _nrm)


print("Avg PSNR:\t",   np.array(psnr_list).mean())
print("Avg FScore:\t", np.array(fsc_list).mean())
print("Avg NRM:\t",    np.array(nrm_list).mean())