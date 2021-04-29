import numpy as np
import win32gui
import cv2
import tkinter as tk
from tkinter import *
from PIL import ImageGrab
from keras.models import load_model

# Import model
model = load_model(r'model/model_chars.h5')

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        # Windows display scale settings
        display_scale = 1.25
        print('test')
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        left, top, right, bottom = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab((left*display_scale, top*display_scale, right*display_scale, bottom*display_scale))

        rgb_image = im.convert('RGB')
        rgb_image.save(r'images/char.jpg')
        # import predict_image

        App.predict_image(self)
        # HWND = self.canvas.winfo_id()  # get the handle of the canvas
        # left, top, right, bottom = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        # if not self.exclude_border:
        #     left, top, right, bottom = win32gui.GetWindowRect(HWND)
        # else:
        # _left, _top, _right, _bottom = win32gui.GetWindowRect(HWND)
        # left, top = win32gui.ClientToScreen(HWND, (_left, _top))
        # right, bottom = win32gui.ClientToScreen(HWND, (_right, _bottom))
        # a, b, c, d = rect
        # rect=(a+15,b,c+10,d+10)
        # rect = (left, top, right, bottom)
        # print(rect)
        # im = ImageGrab.grab((0,0,100,100))
        # rgb_image = im.convert('RGB')
        # rgb_image.save(r'images/char.jpg')

        # digit, acc = App.predict_digit(im)
        # self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')

    def predict_image(self):
        # import cv2
        # import numpy as np
        from keras.models import load_model

        # Import model
        # model = load_model(r'model/model_chars.h5')

        # Dictionary for getting characters from index values...
        word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                     12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                     23: 'X', 24: 'Y', 25: 'Z'}

        # Prediction on external image...
        img = cv2.imread(r'images/char.jpg')
        img_copy = img.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 440))

        img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)
        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

        img_final = cv2.resize(img_thresh, (28, 28))
        img_final = np.reshape(img_final, (1, 28, 28, 1))

        img_pred = word_dict[np.argmax(model.predict(img_final))]
        self.label.configure(text=img_pred + ', ' + str(int(1 * 100)) + '%')

        # cv2.putText(img, "Dataflair _ _ _ ", (20, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color=(0, 0, 230))
        # cv2.putText(img, "Prediction: " + img_pred, (20, 410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color=(255, 0, 30))
        # cv2.imshow('Dataflair handwritten character recognition _ _ _ ', img)

        # while (1):
        #     k = cv2.waitKey(1) & 0xFF
        #     if k == 27:
        #         break
        # cv2.destroyAllWindows()


    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
        print(event.x, event.y)

    # def predict_digit(img):
    #     #resize image to 28x28 pixels
    #     img = img.resize((28,28))
    #     #convert rgb to grayscale
    #     img = img.convert('L')
    #     img = np.array(img)
    #     #reshaping to support our model input and normalizing
    #     img = img.reshape(1,28,28,1)
    #     img = img/255.0
    #     #predicting the class
    #     res = model.predict([img])[0]
    #     return np.argmax(res), max(res)

app = App()
mainloop()

# import predict_image
#
# predict_image