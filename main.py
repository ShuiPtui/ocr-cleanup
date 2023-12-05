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


ret, thresh1 = cv2.threshold(image1, 0, 255, cv2.THRESH_BINARY)
thresh2 = cv2.adaptiveThreshold(image2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print(pt.image_to_string(image1)) ###Image that has black text
print(pt.image_to_string(edges_img2)) ###Image that has stop-sign.jpg. Shouldn't work because text is white
# print(pt.image_to_string(image3))
# print(pt.image_to_string(image4))
# print(pt.image_to_string(image5))
# print(pt.image_to_string(image6))

img = image6.copy()



# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
     
#     # Drawing a rectangle on copied image
#     rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
#     # Cropping the text block for giving input to OCR
#     cropped = img[y:y + h, x:x + w]
     
#     # Apply OCR on the cropped image
#     text = pt.image_to_string(cropped)

#     print(text)