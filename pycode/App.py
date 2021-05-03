import numpy as np
import win32gui
import cv2
import tkinter as tk
import ctypes
from tkinter import *
from PIL import ImageGrab
from keras.models import load_model

# Import model
model = load_model(r'model/model_chars_and_numbs.h5')


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
        # Window display scale settings
        scaleFactor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        left, top, right, bottom = win32gui.GetWindowRect(HWND)
        rect = (left * scaleFactor, top * scaleFactor, right * scaleFactor, bottom * scaleFactor)
        im = ImageGrab.grab(rect)

        rgb_image = im.convert('RGB')
        rgb_image.save(r'images/char.jpg')

        App.predict_image(self)

    def predict_image(self):
        # Dictionary for getting characters from index values...
        word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                     12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                     23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7',
                     34: '8', 35: '9'}

        # Getting drawn image...
        img = cv2.imread(r'images/char.jpg')
        img_copy = img.copy()

        img_copy = cv2.GaussianBlur(img_copy, (7, 7), 0)

        img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

        img_final = cv2.resize(img_thresh, (28, 28))
        img_final = np.reshape(img_final, (1, 28, 28, 1))

        res = model.predict(img_final)[0]

        img_pred = word_dict[np.argmax(res)]
        self.label.configure(text=img_pred + ', ' + str(int(max(res) * 100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
        # print(event.x, event.y)


app = App()
mainloop()
