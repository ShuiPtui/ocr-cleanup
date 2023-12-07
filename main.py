"""
TODO:
        1. Add more buttons to dynamic change image preprocessing
        2. Add a crop function to the interface
        3. Add a button that opens a help window
        4. Add a button that reverts displayed image to original image
        5. Use image_to_data to find boundary boxes as a way to determine text location
        6. Implement text deletion from image
        7. Implement text drawing (PIL or cv2)
        8. Implement some translation API (even chatgpt works)
        9. Perform additional computer photography stuff
        10. Replace existing code with own code (Optional)

"""



import pytesseract as pt
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# import nltk                            #will need to install #replaced by enchant
from spellchecker import SpellChecker   #will need to install
import string
import re
import enchant
import PySimpleGUI as sg
import io

# nltk.download('brown') #downloads required set; slower than nltk.corpus.words but is able to deal with numbers and capitals #Edit: Enchant is so much faster
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


# OCR engines modes:
#   0   Legacy Tesseract only
#   1   Neural nets LSTM (Long Short Term Memory) only
#   2   Legacy Tesseract Combined with LSTM
#   3   Default OCR mode

def image_to_bytes(img):
    pil_image = Image.fromarray(img)
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='PNG')
    img_data = image_bytes.getvalue()
    return img_data

def interface(img):
    h, w = img.shape[:2]
    
    main_layout = create_main_layout(img)
    main_window = sg.Window('OCR Program', main_layout, size=(w+400, h+100), resizable=True)

    invert_toggle = False
    morpho_toggle = False

    while True:
        event, values = main_window.read()

        if event == sg.WINDOW_CLOSED or event == 'EXIT':
            break

        elif event == 'PREPROCESS':
            psmode = int(values['PSM'])
            oemode = int(values['OEM'])
            # custom_config = r'--oem{} --psm{}'.format(oemode, psmode)
            processed_img = image_preprocessing(img, invert_toggle, morpho_toggle)
            main_window['IMAGE'].update(data=image_to_bytes(processed_img))
            h, w = processed_img.shape[:2]
            main_window.size = (w+400, h+100)

    
        elif event == 'INVERT':
            invert_toggle = not invert_toggle
            main_window['INVERT'].update(
                text='Inverted' if invert_toggle else 'Not Inverted', 
                button_color=('black', '#d7c4a4') if invert_toggle else ('white', '#283b5b'))
            
        elif event == 'MORPHO':
            morpho_toggle = not morpho_toggle
            main_window['MORPHO'].update(
                text='Opening' if morpho_toggle else 'Dilation', 
                button_color=('black', '#d7c4a4') if morpho_toggle else ('white', '#283b5b'))

def create_main_layout(img):
    
    img_data = image_to_bytes(img)

    image = sg.Image(
        data= img_data,
        key= 'IMAGE'
    )

    exit_btn = sg.Button(
        button_text= 'Exit',
        key= 'EXIT'
    )

    perform_preprocessing = sg.Button(
        button_text='Perform Preprocessing',
        key='PREPROCESS'
    )

    invert = sg.Button(
        button_text= 'Not Inverted',
        key= 'INVERT',
        enable_events=True
    )

    morpho_transformation = sg.Button(
        button_text= 'Dilation',
        key= 'MORPHO',
        enable_events=True
    )

    psm_slider = sg.Slider(
        range=(1,12),
        size=(30, 15),
        orientation='h',
        default_value=6,
        key='PSM'
    )

    psm = sg.Text('PSM')

    oem_slider = sg.Slider(
        range=(0,3),
        size=(30, 15),
        orientation='h',
        default_value=3,
        key='OEM'
    )

    oem = sg.Text('OEM')

    left_col = sg.Column([
        [image],
        [exit_btn]
    ])

    options_col = sg.Column([
        [psm, psm_slider],
        [oem, oem_slider],
        [invert, morpho_transformation],
        [perform_preprocessing]
    ])

    

    layout = [
        [left_col, options_col]
    ]

    return layout


def display_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey()

def image_preprocessing(img, invert_state=False, morpho_state=False):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1000, int(h * (1000 / float(w)))), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    blurred_img1 = cv2.GaussianBlur(img, (3,3), 0)
    # edges_img1 = cv2.Canny(blurred_img1, 50, 150) #shows edges of the image
    if invert_state:
        thresh = cv2.threshold(blurred_img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] #threshold attributes may need to be changed
    else:
        thresh = cv2.threshold(blurred_img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    #perform a morphological transformation. We could add more variety depending on the task at hand
    if morpho_state:
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1) #Good for thick text
        invert = 255 - opening
    else:
        dilation = cv2.dilate(thresh, kernel, iterations = 1) #Good for thin text
        #invert the image (we want the text to be black and foreground to be white)
        invert = 255 - dilation

    return invert

def image_ocr(img):
    custom_config = r'--oem 3 --psm 6'
    results = pt.image_to_string(img, lang='eng', config=custom_config)
    data = pt.image_to_data(img, lang='eng', config=custom_config, output_type=pt.Output.DICT)
    print(data)
    return results

def check_results(text):
    spell = SpellChecker()
    corrected_line = ''
    new_lines = []
    split_pattern = re.compile(r'[' + re.escape(string.punctuation) + r'\s]+')
    valid = False
    special_condition = False #for when word correcter is unable to handle the text and returns i

    #This could needs to be optimized
    for line in text.splitlines():
        if not line == '':
            
            for word in split_pattern.split(line):
                if len(corrected_line) > 0 and valid == True and special_condition == False:
                    corrected_line += ' '
                    valid = False
                    special_condition = False

                if not word == '':
                    corrected = word_correction(spell, word)
                    if not corrected == '':
                        valid = valid_word(corrected)
                        if valid:
                            print(word + ' ' + corrected)
                            if not corrected == 'i':
                                corrected_line += corrected.lower()
                            else:
                                if corrected == word:
                                    corrected_line += corrected.lower()
                                else:
                                    special_condition = True

            if valid: 
                new_lines.append(corrected_line)
                corrected_line = ''
                
    return new_lines

def valid_word(word):
    eng_dict = enchant.Dict('en_US')
    if not word in string.punctuation or word.isspace():
        return eng_dict.check(word)
    else:
        return False
    # return word.lower() in set(nltk.corpus.brown.words()) #in my opinion, it takes too long for practical use

def word_correction(spell, word):
    return spell.correction(word)
    
#test image 'coffee store.jpg' works well now
#stop sign image doesn't work with current implementation; likely due to custom config psm

image = cv2.imread('text.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image_preprocessing(image)
# text = image_ocr(image)
# display_image(image)

# results = check_results(text) 
# print(results)
interface(image)