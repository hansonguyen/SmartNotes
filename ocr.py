import os
import json
import cv2 as cv
import numpy as np
from google.cloud import vision
from google.cloud.vision_v1 import types
from typing import Dict, Any

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'GCP_Service_Acct.json'

# Remove noise from image
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
    image = cv.medianBlur(image, 3)
    return image

# Uniform stroke widths
def thinning_image(image):
    image = cv.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.bitwise_not(image)
    return image

"""
Returns path of processed image ("./images/[filename]_processed.[filetype]")
Preprocess image in multiple steps:
1. Binarization
2. Noise Removal
3. Thinning/Skeletonization
"""
def image_preprocess(path: str) -> str:
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # 1. Binarization
    img = cv.medianBlur(img, 5)
    ret,th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # 2. Noise removal
    th1 = noise_removal(th1)
    # 3. Thinning/Skeletonization
    th1 = thinning_image(th1)
    updatedPath = path.split('/')[0] + '/' + path.split('/')[1] + '/' + path.split('/')[2].split('.')[0] + "_processed." + path.split('/')[2].split('.')[1]
    cv.imwrite(updatedPath, th1)
    return updatedPath

"""
Returns JSON response of image
Reads image from images folder and extracts text into JSON
"""
def detect_document(filename: str) -> str:
    # Detects document features in an image.
    print("Scanning image...")
    client = vision.ImageAnnotatorClient()

    # Preprocess image -> use path of processed image
    path = './images/' + filename
    if not os.path.isfile(path):
        raise OSError("Invalid filename: File must exist in images directory")
    path = image_preprocess(path)

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    # Error handling
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    # Convert response to JSON format
    json_response = response_to_json(response, path)

    # Write JSON to a file
    output_file = "./outputs/cloud_vision_output.json"
    with open(output_file, 'w') as json_file:
        json.dump(json_response, json_file, indent=2)
    print("Finished scanning image")
    return json_response["full_text_annotation"][0]["text"]

"""
Returns Cloud Vision response as JSON
"""
def response_to_json(response, path: str) -> Dict[Any, Any]:
    json_response = {}
    json_response['full_text_annotation'] = []
    for page in response.full_text_annotation.pages:
        page_data = {
            'file_name': path.split('/')[2],
            'width': page.width,
            'height': page.height,
            'text': '',
            'blocks': []
        }
        for block in page.blocks:
            block_data = {
                'block_type': block.block_type,
                'confidence': block.confidence,
                'text': '',
                'paragraphs': []
            }
            for paragraph in block.paragraphs:
                paragraph_data = {
                    'text': '',
                    'confidence': paragraph.confidence,
                    'words': []
                }
                for word in paragraph.words:
                    word_text = ''.join(
                        [symbol.text for symbol in word.symbols])
                    word_data = {
                        'text': word_text,
                        'confidence': word.confidence
                    }
                    paragraph_data['words'].append(word_data)
                paragraph_text = ' '.join([word['text']
                                          for word in paragraph_data['words']])
                paragraph_data['text'] = paragraph_text
                block_data['paragraphs'].append(paragraph_data)
            block_text = ' '.join([paragraph['text']
                                  for paragraph in block_data['paragraphs']])
            block_data['text'] = block_text
            page_data['text'] += block_text + '\n'
            page_data['blocks'].append(block_data)
        json_response['full_text_annotation'].append(page_data)

    return json_response