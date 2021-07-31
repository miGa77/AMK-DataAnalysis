import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# import Labels from mapping.txt
def __import_mapping_file(filename):
    with open(filename) as file:
        content = file.read().splitlines()
        labels = []
        for line in content:
            arr = line.split()
            labels.insert(int(arr[0]), chr(int(arr[1])))
    return labels


__LABELS = __import_mapping_file("data/emnist-balanced-mapping.txt")


# region Library
def recognize(filename, model):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        try:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            top = int(0.025 * thresh.shape[0])
            bottom = top
            left = int(0.025 * thresh.shape[1])
            right = left
            cv2.copyMakeBorder(thresh, top, bottom, left, right, cv2.BORDER_REPLICATE)
            roi = thresh[y - top:y + h + bottom, x - left:x + w + right]
            img = cv2.resize(roi, (28, 28), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            img = img.reshape(1, 28, 28, 1)
            img = img / 255.0
            pred = model.predict([img])[0]
            index = np.argmax(pred)
            accuracy = str(round(np.max(pred), 2))
            final_pred = __LABELS[index]
            data = str(final_pred)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            output_text = "Char: " + data + ", Accuracy: " + accuracy
            cv2.putText(image, output_text, (x, y - 5), font, fontScale, color, thickness)
            cv2.imshow('image', image)
            cv2.waitKey(0)
        except Exception as e:
            print(str(e))


def resize_picture_to_useful_format(filename):
    image = Image.open(filename, 'r')
    width = image.size[0]
    height = image.size[1]
    resize_width = 28 - ((width % 28)) + width + (3 * 28)
    resize_height = (28 - (height % 28)) + height + (3 * 28)
    resized_image = Image.new('RGB', (resize_width, resize_height), (255, 255, 255))
    offset = (int(round(((resize_width - width) / 2), 0)), int(round(((resize_height - height) / 2), 0)))
    resized_image.paste(image, offset)
    file_name, file_extension = os.path.splitext(filename)
    save_filename = file_name + "_resized" + file_extension
    resized_image.save(save_filename)
    return save_filename

# endregion
