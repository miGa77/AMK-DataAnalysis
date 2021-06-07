import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tkinter as tk
from tkinter import *

from PIL import ImageGrab
from keras.models import load_model
import evalPicture as ownLib

model = load_model(r'model_Balanced.h5')
width = 952
height = 952
filename = r'images/my_drawing.jpg'


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.resizable(width=False, height=False)
        self.title("Handwritten Expressions Recognition")
        # Creating elements
        self.canvas = tk.Canvas(self, width=width, height=height, bg="white", cursor="cross")
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.classify_btn.grid(row=0, column=1, pady=20, padx=20)
        self.button_clear.grid(row=0, column=2, pady=20, padx=20)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        # font weight
        r = 6
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

    def classify_handwriting(self):
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
        ownLib.recognize(filename, model)


app = App()
tk.mainloop()
