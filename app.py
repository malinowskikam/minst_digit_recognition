from tkinter import *
from PIL import ImageGrab
import cv2
from keras.engine.saving import load_model
from numpy import argmax

canvas_width =280
canvas_height = 280

interface_scaling = 1.25 #win10 interface scaling


def paint(event):
    color = "#000000"
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    w.create_oval(x1, y1, x2, y2, fill=color)


def predict(event):
    master.update_idletasks()
    x2 = master.winfo_rootx() + w.winfo_x()
    y2 = master.winfo_rooty() + w.winfo_y()
    x1 = x2 + w.winfo_width()
    y1 = y2 + w.winfo_height()
    ImageGrab.grab().crop((x2*interface_scaling, y2*interface_scaling, x1*interface_scaling, y1*interface_scaling)).save("temp.jpg")
    img = cv2.imread("temp.jpg",cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28), interpolation=cv2.INTER_CUBIC)

    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] = 255 - img[i][j]

    img = img.reshape(-1,28,28,1)
    prediction = model.predict(img)
    index = argmax(prediction)
    print(f"{index} - {prediction[0][index]*100}%")


def clean_screen(event):
    w.delete("all")


model = load_model("models/convolutional_neural_network_30_86.h5")
master = Tk()
master.geometry("280x280")
master.title("App")


w = Canvas(master,
           width=canvas_width,
           height=canvas_height,
           bd=0,
           highlightthickness=0,
           relief='ridge')

w.pack()
w.configure(bg="#FFFFFF")
w.bind("<B1-Motion>", paint)
w.bind("<space>", clean_screen)
w.bind("<Return>", predict)
w.focus_set()

mainloop()