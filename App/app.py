from tkinter import *
from PIL import ImageGrab
from keras.models import load_model
import evalPicture as lib


# WORK IN PROGRESS
# region Funktionen
def clear_widget():
    global cv
    # To clear a canvas
    cv.delete("all")


def activate_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y


def draw_lines(event):
    global lasty, lasty
    x, y = event.x, event.y
    # do the canvas drawings
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE)
    lasty, lasty = x, y


def Recognize_Digit():
    global image_number
    # image_number = 0
    filename = f'image_{image_number}.png'
    widget = cv

    # get the widget coordinates
    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    # grab the image, crop it according to the requirement and saved it in png format
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    lib.recognize(filename, model)


# endregion

# region Main
# load model
model = load_model(r'model_Balanced.h5')
print("Model load successfully, go for the APP")
# create a main window first(names as root)
root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recognition GUI App")

# Initialize few variables
lastx, lasty = None, None
image_number = 0

# create canvas for drawing
cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
cv.bind('<Button-1>', activate_event)

# Add Buttons and Labels
btn_save = Button(text='Recognize Digit', command=Recognize_Digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text="Clear Widget", command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

# mainloop
root.mainloop()

# endregion
