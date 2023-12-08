"""
TODO:
        1. Add more buttons to dynamic change image preprocessing (optional; current methods should be good enough for basic stuff)
        2. Add a crop function to the interface (optional; user can just provide cropped images)
        
  
        
        9. Perform additional computer photography stuff (perhaps making special fonts or giving replaced text some effect)
        10. Replace existing code with own code (Optional)
        

Current issues:
        - None so far
        

Fixed issues:
        - Currently, widths are increased to crazy sizes, possible culprit is the method to deal with
        way too long text 
        (EDIT: I believe the error is caused by how I handle the excess text; fixing
        the way it saves the excess text may prevent the error.)
        (EDIT 2: Pretty sure the error is how I am saving the excess text)
        (EDIT 3: Fixed, it was due to how I was storing excess string)
        - Font size needs to be dynamically generated to either best fit the boundary or perform some
        lazy method to resize it (not best fit)

COMPLETED:
        3. Add a button that opens a help window
        4. Add a button that reverts displayed image to original image
        5. ***IMPORTANT*** Use image_to_data to find boundary boxes as a way to determine text location  ***IMPORTANT***
        6. ***IMPORTANT*** Implement text deletion from image ***IMPORTANT***
        7. ***IMPORTANT*** Implement text drawing (PIL or cv2) ***IMPORTANT***
        8. Implement some translation API (even chatgpt works)
        11. Implement another button to handle OCR and text replacement 
"""



import pytesseract as pt
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
# import nltk                            #will need to install #replaced by enchant
from spellchecker import SpellChecker   #will need to install
import string
import re
import enchant
import PySimpleGUI as sg
import io
from googletrans import Translator

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

def translate_text(text, target='fr'):
    translator = Translator()
    translation = translator.translate(text, dest=target)
    return translation.text

def image_to_bytes(img):
    pil_image = Image.fromarray(img)
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='PNG')
    img_data = image_bytes.getvalue()
    return img_data

def interface(img):
    original_img = img.copy()
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
            custom_config = r'--oem {} --psm {}'.format(oemode, psmode)
            processed_img = image_preprocessing(img, invert_toggle, morpho_toggle)
            # results = image_ocr(processed_img, custom_config)
            # text_removed_img, positions = text_removal(img, invert_toggle)
            # corrections = check_results(results)
            
            # pil_copy = Image.fromarray(text_removed_img)
            # text_adder(corrections, pil_copy, positions)
            

            main_window['IMAGE'].update(data=image_to_bytes(processed_img))
            h, w = processed_img.shape[:2]
            main_window.size = (w+400, h+100)
        
        elif event == 'TRANSLATE':
            #Perform the following if we assume that they do not perform processing
            psmode = int(values['PSM'])
            oemode = int(values['OEM'])
            custom_config = r'--oem {} --psm {}'.format(oemode, psmode)
            processed_img = image_preprocessing(img, invert_toggle, morpho_toggle)
            results = image_ocr(processed_img, custom_config)
            text_removed_img, positions = text_removal(img, invert_toggle)
            corrections = check_results(results)
            
            pil_copy = Image.fromarray(text_removed_img)
            text_adder(corrections, pil_copy, positions)
            
            main_window['IMAGE'].update(data=image_to_bytes(np.array(pil_copy)))
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
        elif event == 'HELP':
            # Open a new window with help information
            sg.popup('Help ', 'Use the buttons to preprocess the image and perform OCR.', title='Help')

        elif event == 'REVERT':
            # Revert to the original image
            main_window['IMAGE'].update(data=image_to_bytes(original_img))
            h, w = original_img.shape[:2]
            main_window.size = (w + 400, h + 100)
        

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

    translate = sg.Button(
        button_text='Translate Image',
        key='TRANSLATE'
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

    help_btn = sg.Button(
        button_text='Help',
        key='HELP'
    )

    revert_btn = sg.Button(
        button_text='Revert to Original',
        key='REVERT'
    )

    options_col = sg.Column([
        [psm, psm_slider],
        [oem, oem_slider],
        [invert, morpho_transformation],
        [perform_preprocessing, translate],
        [help_btn, revert_btn]
    ])

    left_col = sg.Column([
        [image],
        [exit_btn]
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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    blurred_img1 = cv2.GaussianBlur(gray, (3,3), 0)
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
        # invert = 255 - dilation
        invert = 255 - dilation

    return invert

def text_removal(img, invert_state=False):
    h, w = img.shape[:2]
    img = cv2.resize(img, (1000, int(h * (1000 / float(w)))), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray, (3,3), 0)
    if invert_state:
        thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] #threshold attributes may need to be changed
    else:
        thresh = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1)) #values affect the removal

    #perform a morphological transformation. We could add more variety depending on the task at hand
    dilation = cv2.dilate(close, dilation_kernel, iterations = 1)

    cnts = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    positions = []
    i = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 800 and area < 15000:
            x,y,w,h = cv2.boundingRect(c)
            i+=1
            print(i)
            
            positions.append([x, y, w, h])
            # cv2.imshow('image', img[y:y+h, x:x+w])
            # cv2.waitKey()
            cv2.rectangle(img, (x, y), (x + w, y + h), (52, 53, 65), -1) #we can find dominant color using k-means or perform image reconstruction

    # cv2.imshow('image', img)
    # cv2.waitKey()
    return img, positions

def text_adder(text, img, positions):
    print(text)
    # text = text[::-1]
    print(text)
    print(positions)
    positions = positions[::-1]
    print(positions)
    draw = ImageDraw.Draw(img)
    
    excess = ''
    # print(len(text))
    for i in range(len(text)):
        h = positions[-i][3]
        # print('This is text height: {}'.format(h))
        font = '/usr/share/fonts/truetype/Nakula/nakula.ttf'
        current_h = h
        text_font = ImageFont.truetype('/usr/share/fonts/truetype/Nakula/nakula.ttf', h) #Make this more dynamic
        # text_font = ImageFont.truetype('/usr/share/fonts/truetype/Nakula/nakula.ttf', 6) #temporary size
        line = translate_text(text[i])
        
        if len(excess) > 0:
            line = excess + ' ' + line
            excess = ''

        text_width = draw.textlength(line, font=text_font)
        max_width = positions[i][2]

        start_point = (positions[i][0], positions[i][1]-10)


        if text_width > max_width:
            # print('exception')
            # Adjust the text and get the excess text
            adjusted_text, excess_text, new_font = resize_text(line, font, current_h, max_width)
            text_font = new_font
            excess = excess_text
            
            # Draw the adjusted text on the image
            draw.text(start_point, adjusted_text, font=text_font, fill=(255, 255, 255))  # White text color

            # # Return the excess text for further processing
            
        else:
            # Draw the original text on the image
            draw.text(start_point, line, font=text_font, fill=(255, 255, 255))  # White text color


def fix_text_length(text, text_font, max_width):
    draw = ImageDraw.Draw(Image.new('RGB', (1,1)))
    text_width = draw.textlength(text, font=text_font)
    original = text
    excess_text = ''
    # print(max_width)
    while text_width > max_width and len(text) > 0:

        text = text[:-1]
        text_width = draw.textlength(text, font=text_font)
    
    excess_text = original[len(text):]
    # print(excess_text)

    return text, excess_text

#alternative method to handle really long lines
def resize_text(text, font, current_h, max_width):

    draw = ImageDraw.Draw(Image.new('RGB', (1,1)))


    text_font = ImageFont.truetype(font, current_h)
    text_width = draw.textlength(text, font=text_font)

    while text_width > max_width and current_h > 5:
        current_h -=1
        text_font = ImageFont.truetype(font, current_h)
        text_width = draw.textlength(text, font=text_font)

    if current_h <= 5:
        text, excess = fix_text_length(text, text_font, max_width)
        return text, excess, text_font

    return text, '', text_font

def image_ocr(img, config=r'--oem 3 --psm 6'):

    results = pt.image_to_string(img, lang='eng', config=config)
    data = pt.image_to_data(img, lang='eng', config=config, output_type=pt.Output.DICT)
    # print(results)

    num_words = len(data['text'])
    # for n in range(num_words):
    #     if data['text'][n].strip():
    #         left = data['left'][n]
    #         top = data['top'][n]
    #         width = data['width'][n]
    #         height = data['height'][n]
    #         print(f"Bounding Box {n + 1}: Left={left}, Top={top}, Width={width}, Height={height}, Text={data['text'][n]}")



    return results

def check_results(text):
    spell = SpellChecker()
    corrected_line = ''
    new_lines = []
    #split_pattern = re.compile(r'[' + re.escape(string.punctuation) + r'\s]+')
    split_pattern = re.compile(r'[' + r'\s]+')
    valid = False

    #This could needs to be optimized
    for line in text.splitlines():
        # print(valid)
        # print(special_condition)
        if not line == '':
            
            for word in split_pattern.split(line):
                if len(corrected_line) > 0 and not word == '':
                    corrected_line += ' '
                    valid = False
                

                if not word == '':
                    corrected = word_correction(spell, word)
                    if corrected == None:
                        corrected = word
                        corrected_line += corrected.lower()
                    
                    if not corrected == '' and not corrected == 'i':
                        valid = valid_word(corrected)
                        if valid:
                            # print(word + ' ' + corrected)
                        
                            corrected_line += corrected.lower()
                            
                    elif not corrected == '':
                        if corrected == word:
                            corrected_line += corrected.lower()
                                

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