from pytesseract import image_to_string
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
# %matplotlib inline

def img2str(img):
  print('image to string:\n----------------')
  text = image_to_string(img)
  print(text)

def plot_img(images, titles):
  fig, axs = plt.subplots(nrows = 1, ncols = len(images), figsize = (15, 15))
  for i, p in enumerate(images):
    axs[i].imshow(p, 'gray')
    axs[i].set_title(titles[i])
    #axs[i].axis('off')
  plt.show()

# 0 membaca citra dalam bentuk grayscale
img1 = cv2.imread('steve.webp', 0)
plt.imshow(img1, 'gray')
plt.show()
img2str(img1)

# crop img1
crop_img1 = img1[130:600,100:730]
plt.imshow(crop_img1,'gray')
plt.show()
img2str(crop_img1)

# menampilkan histogram dari citra
plt.hist(crop_img1.ravel(),256,[0,256]) 
plt.show() 

# global thresholding
ret, img_binary = cv2.threshold(crop_img1, 127, 255, cv2.THRESH_BINARY)
ret, img_binary_inv = cv2.threshold(crop_img1, 127, 255, cv2.THRESH_BINARY_INV)
ret, img_threshtozero = cv2.threshold(crop_img1, 127, 255, cv2.THRESH_TOZERO)
ret, img_threshtozero_inv = cv2.threshold(crop_img1, 127, 255, cv2.THRESH_TOZERO_INV)
ret, img_thresh_trunc = cv2.threshold(crop_img1, 127, 255, cv2.THRESH_TRUNC)

# plot citra global tresholding
images = [img_binary, img_binary_inv, img_threshtozero, img_threshtozero_inv, img_thresh_trunc]
titles = ['binary', 'binary_inv', 'tozero', 'tozero_inv', 'trunc']
plot_img(images, titles)

# menampilkan hasil string dari global tresholding
print('binary')
img2str(img_binary)
print('binary_inv')
img2str(img_binary_inv)
print('tozero')
img2str(img_threshtozero)
print('tozero_inv')
img2str(img_threshtozero_inv)
