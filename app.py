from tkinter import *
from PIL import ImageGrab
import cv2
from keras.engine.saving import load_model
from numpy import argmax

canvas_width =281
canvas_height = 281


def paint(event):
    color = "#000000"
    x1, y1 = (event.x - 4), (event.y - 4)
    x2, y2 = (event.x + 4), (event.y + 4)
    w.create_oval(x1, y1, x2, y2, fill=color)


def predict(event):
    x2 = master.winfo_rootx() + w.winfo_x() + 2
    y2 = master.winfo_rooty() + w.winfo_y() + 2
    x1 = x2 + w.winfo_width() - 5
    y1 = y2 + w.winfo_height() - 5
    ImageGrab.grab().crop((x2, y2, x1, y1)).save("temp.jpg")
    img = cv2.imread("temp.jpg",cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)

    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = 255 - img[i][j]

    img = img.reshape(-1,28,28,1)
    print(argmax(model.predict(img)))



def clean_screen(event):
    w.delete("all")

model = load_model("models/convolutional_neural_network_10_86.h5")
master = Tk()
master.title("Test")


w = Canvas(master,
           width=canvas_width,
           height=canvas_height)
w.pack(expand=YES, fill=BOTH)
w.configure(bg="#FFFFFF")
w.bind("<B1-Motion>", paint)
w.bind("<space>", clean_screen)
w.bind("<Return>", predict)
w.focus_set()

mainloop()