# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:52:33 2019

@author: hp
"""

#ui.py
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk

window = tk.Tk()

window.title("Dr.RicePlant")

window.geometry("500x510")
window.configure(background ="lightgreen")

title = tk.Label(text="Click below to choose picture for testing disease....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()
def bact():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. RicePlant")

    window1.geometry("500x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for BrownSpot are:\n\n "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = "Improving soil fertility.\nMonitor soil nutrients regularly.\n Apply required fertilizers.\nFor soils that are low in silicon,\napply calcium silicate slag before planting"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                        fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def vir():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr. RicePlant")

    window1.geometry("500x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for Hispa disease are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = "Avoid over fertilizing the field.\nTo prevent egg laying of the pests,the shoot tips can be cut.\nClipping and burying shoots in the mud can reduce grub populations.\nAvoid excessive nitrogen fertilization in infested fields"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=30, pady=30)

    window1.mainloop()

def latebl():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Dr.RicePlant")

    window1.geometry("500x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()
    rem = "The remedies for Rice LeafBlast are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                      fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = "Burn previously blast affected straw and stubbles.\nUse of disease free seeds.\nSeed treatment with Pseudomonas fluorescence 10g / 1 of water for 30 min."
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm   
    verify_dir = "C:\\Users\\hp\\.spyder-py3\\Dataset\\testpic"
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'Healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'BrownSpot'
        elif np.argmax(model_out) == 2:
            str_label = 'Hispa'
        elif np.argmax(model_out) == 3:
            str_label = 'LeafBlast'

        if str_label =='Healthy':
            status ="HEALTHY"
        else:
            status = "UNHEALTHY"

        message = tk.Label(text='Status: '+status, background="lightgreen",
                           fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)
        if str_label == 'BrownSpot':
            diseasename = "Brown Spot "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=bact)
            button3.grid(column=0, row=6, padx=10, pady=10)
        elif str_label == 'Hispa':
            diseasename = "Hispa Disease "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=vir)
            button3.grid(column=0, row=6, padx=10, pady=10)
        elif str_label == 'LeafBlast':
            diseasename = "Leaf Blast "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=latebl)
            button3.grid(column=0, row=6, padx=10, pady=10)
        else:
            r = tk.Label(text='Plant is healthy', background="lightgreen", fg="Black",
                         font=("", 15))
            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=30, pady=30)

def openphoto():
    dirPath = "C:\\Users\\hp\\.spyder-py3\\Dataset\\testpic"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/hp/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='C:\\Users\\hp\\.spyder-py3\\Dataset\\', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "C:\\Users\\hp\\.spyder-py3\\Dataset\\testpic"
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="220", width="490")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)
window.mainloop()



