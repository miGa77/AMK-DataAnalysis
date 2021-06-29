import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import *

from PIL import ImageGrab
from keras.models import load_model
import evalPicture as ownLib

model = load_model(r'trainedModel.h5')
width = 748
height = 748
filename = r'images/my_drawing.png'


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # font weight
        self.font_weight = 7
        self.resizable(width=False, height=False)
        self.title("Handwritten Expressions Recognition")
        # Creating elements
        self.canvas = tk.Canvas(self, width=width, height=height, bg="white", cursor="cross")
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.button_toggle_font_weight = tk.Button(self, text="Toggle Font Weight", command=self.toggle)
        # Grid structure
        self.canvas.grid(row=0, pady=2, column=0, sticky=W)
        self.classify_btn.grid(row=0, column=1, pady=20, padx=20)
        self.button_clear.grid(row=0, column=2, pady=20, padx=20)
        self.button_toggle_font_weight.grid(row=0, column=3, pady=20, padx=5)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def toggle(self):
        if self.font_weight == 7:
            self.font_weight = 13
        else:
            self.font_weight = 7

    def clear_all(self):
        self.canvas.delete("all")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        self.canvas.create_oval(self.x - self.font_weight, self.y - self.font_weight, self.x + self.font_weight,
                                self.y + self.font_weight, fill='black')

    def classify_handwriting(self):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        ImageGrab.grab().crop((x + 3, y + 3, x1 - 3, y1 - 3)).save(filename)
        resized_filename = ownLib.resize_picture_to_useful_format(filename)
        ownLib.recognize(resized_filename, model)


app = App()
tk.mainloop()
