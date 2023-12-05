import pytesseract as pt
import cv2
from PIL import Image

#Apparently pytesseract works best with PIL


image1 = cv2.imread('black text.png')
image2 = cv2.imread('stop-sign.jpg')
# imagePIL2 = Image.open('stop-sign.jpg')
img_PIL2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY))
image3 = cv2.imread('store.jpeg')
image4 = cv2.imread('drawing.png')
image5 = cv2.imread('coffee store.jpg')
image6 = cv2.imread('drawing with noise.png')

image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
image5 = cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)
image6 = cv2.cvtColor(image6, cv2.COLOR_BGR2GRAY)


blurred_img2 = cv2.GaussianBlur(image2, (5,5), 0)
edges_img2 = cv2.Canny(blurred_img2, 50, 150)
ret, thresh1 = cv2.threshold(edges_img2, 127, 255, cv2.THRESH_BINARY)


cv2.imshow('image', thresh1) 
  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 

# print(pt.image_to_string(image1)) ###Image that has black text
print(pt.image_to_string(thresh1, lang='eng', config='--psm 6')) #Stop is read when I changed the config. I presume that the config is important
# print(pt.image_to_string(image3))
# print(pt.image_to_string(image4))
# print(pt.image_to_string(image5))
# print(pt.image_to_string(image6))



