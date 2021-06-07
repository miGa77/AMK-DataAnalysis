import cv2
import imutils
import numpy as np

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
          'f', 'g', 'h', 'n', 'q', 'r', 't']


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
            final_pred = LABELS[index]
            data = str(final_pred)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)
            cv2.imshow('image', image)
            cv2.waitKey(0)
        except Exception as e:
            print(str(e))

# endregion
