import pytesseract as pt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import nltk                            #will need to install
from spellchecker import SpellChecker   #will need to install
import string
import re

nltk.download('brown') #downloads required set; slower than nltk.corpus.words but is able to deal with numbers and capitals
#include 'pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>' if you don't have tesseract in your PATH

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

def display_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey()

def image_preprocessing(img):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1000, int(h * (1000 / float(w)))), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred_img1 = cv2.GaussianBlur(img, (3,3), 0)
    # edges_img1 = cv2.Canny(blurred_img1, 50, 150) #shows edges of the image
    thresh = cv2.threshold(blurred_img1, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)[1] #threshold attributes may need to be changed

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    #perform a morphological transformation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    dilation = cv2.dilate(thresh,kernel,iterations = 1)

    #invert the image
    invert = 255 - opening

    return invert

def image_ocr(img):
    custom_config = r'--oem 3 --psm 6'
    results = pt.image_to_string(img, lang='eng', config=custom_config)
    print(results)
    return results

def check_results(text):
    spell = SpellChecker()
    corrected_line = ''
    new_lines = []
    split_pattern = re.compile(r'[' + re.escape(string.punctuation) + r'\s]+')

    for line in text.splitlines():
        for word in split_pattern.split(line):
            
            corrected = word_correction(spell, word)
            print(word + ' ' + corrected)
            if (valid_word(corrected)):
                corrected_line += corrected.lower()
                corrected_line += ' '
        new_lines.append(corrected_line)
        corrected_line = ''

    return new_lines

def valid_word(word):
    return word.lower() in set(nltk.corpus.brown.words())

def word_correction(spell, word):
    if word.isspace() or word == '':
        return word
    else:
        return spell.correction(word)
    

image = cv2.imread('coffee store.jpg')
image = image_preprocessing(image)
text = image_ocr(image)
display_image(image)

results = check_results(text) 
print(results)