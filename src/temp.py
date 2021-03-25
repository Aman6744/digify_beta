import matplotlib.pyplot as plt 
import cv2

img1 = cv2.imread("img1.png",0)
img2 = cv2.imread("img2.png",0)
plt.subplot(211)
plt.imshow(img1, cmap="gray")
plt.subplot(212)
plt.imshow(img2, cmap="gray")
plt.show()