import pytesseract as pt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#Apparently pytesseract works best with PIL


image1 = cv2.imread('black text.png')
image2 = cv2.imread('stop-sign.jpg')
imagePIL2 = Image.open('stop-sign.jpg')
img_PIL2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY))
image3 = cv2.imread('store.jpeg')
image4 = cv2.imread('drawing.png')
image5a = cv2.imread('coffee store.jpg')
image5b = cv2.imread('coffee store cropped.png') #If an image is cropped well, text detection of OCR works much better
image5c = cv2.imread('coffeestorecropped.png') #has added metadata
image6 = cv2.imread('drawing with noise.png')

image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
image5 = cv2.cvtColor(image5a, cv2.COLOR_BGR2GRAY)
image6 = cv2.cvtColor(image6, cv2.COLOR_BGR2GRAY)

#image processing before OCR to help with detecting text
h, w = image5.shape[:2]
image5 = cv2.resize(image5, (1000, int(h * (1000 / float(w)))))
blurred_img1 = cv2.GaussianBlur(image5, (5,5), 0)
edges_img1 = cv2.Canny(blurred_img1, 50, 150)
ret, thresh1 = cv2.threshold(edges_img1, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#pytesseract.image_to_boxes returns pretty useful information for replacing text 
# <character> <left> <bottom> <right> <top> <confidence>


boxes = pt.image_to_boxes(thresh1)
# Parse bounding box coordinates
bounding_boxes = [line.split()[1:5] for line in boxes.splitlines()]

# Convert bounding box coordinates to integers
bounding_boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in bounding_boxes]

# Calculate the crop region based on bounding boxes
min_x = min(box[0] for box in bounding_boxes)
min_y = min(box[1] for box in bounding_boxes)
max_x = max(box[2] for box in bounding_boxes)
max_y = max(box[3] for box in bounding_boxes)


print('Min x: ' + str(min_x))
print('Min y: ' + str(min_y))
print('Max x: ' + str(max_x))
print('Max y: ' + str(max_y))


#pytesseract.image_to_data can provide more info
data = pt.image_to_data(thresh1, output_type=pt.Output.DICT, config='--psm 11 --oem 3')
num_boxes = len(data['level'])
# for i in range(num_boxes):
#     confidence = data['conf'][i]
#     if confidence >= 30:
#         print(confidence)
#         (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#         crop = image5[y:y+h, x:x+w]
#         pil_image = Image.fromarray(crop)

#         blurred_img2 = cv2.GaussianBlur(np.array(pil_image), (5,5), 0)
#         edges_img2 = cv2.Canny(blurred_img2, 50, 150)
#         ret, thresh2 = cv2.threshold(edges_img2, 0, 255, cv2.THRESH_BINARY)



#         print(pt.image_to_string(thresh2, lang='eng', config='--oem 3 --psm 8')) #Stop is read when I changed the config. I presume that the config is important
#         plt.imshow(thresh2)
#         plt.axis('off')  # Turn off axis labels
#         plt.show()
#         cv2.rectangle(image5a, (x, y), (x + w, y + h), (0, 255, 0), 2) #show boundary boxes


# print('Data info: {}'.format(data))

cv2.imshow('image', image5a) 


  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  

# closing all open windows 
cv2.destroyAllWindows() 

print(pt.image_to_string(thresh1, lang='eng', config='--psm 11')) #Stop is read when I changed the config. I presume that the config is important


# Page segmentation modes:
#   0    Orientation and script detection (OSD) only.
#   1    Automatic page segmentation with OSD.
#   2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
#   3    Fully automatic page segmentation, but no OSD. (Default)
#   4    Assume a single column of text of variable sizes.
#   5    Assume a single uniform block of vertically aligned text.
#   6    Assume a single uniform block of text.
#   7    Treat the image as a single text line.
#   8    Treat the image as a single word.
#   9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
#        bypassing hacks that are Tesseract-specific.



