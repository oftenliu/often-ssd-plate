
import os
import math
import random

import numpy as np
import cv2

# 举个栗子，添加输入框，将验证码图片打印出来
# coding= utf-8

from PIL import ImageTk
from tkinter import *
import PIL
import tkinter as tk
import os

def traverse(f,imageset):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            ext = os.path.splitext(tmp_path)[-1][1:]
            if ext == "jpg" or ext == "png" or ext == "bmp":
                #print('文件: %s'%tmp_path)
                imageset.append(tmp_path)
        else:
            print('文件夹：%s'%tmp_path)
            traverse(tmp_path,imageset)

class GetCode(object):

    def __init__(self):

        self.root = tk.Tk()
        self.root.geometry('1280x720') 
        self.root.resizable(width=False,height=False)   # 固定长宽不可拉伸
        self.dataset = []
        path = '../result/'
        traverse(path,self.dataset)
        self.index = 0
        self.im=PIL.Image.open(self.dataset[0])
        self.img=ImageTk.PhotoImage(self.im)
        self.imLabel=tk.Label(self.root,image=self.img,width=720,height=640) # 显示图片
        self.imLabel.pack(side=LEFT)
        self.but1 = tk.Button(self.root,text="ok",command=self.ok_fn).pack(side=LEFT) # 按键
        self.but1 = tk.Button(self.root,text="error",command=self.error_fn).pack(side=LEFT) # 按键
        self.but1 = tk.Button(self.root,text="none",command=self.none_fn).pack(side=LEFT) # 按键
        self.root.mainloop()

    def ok_fn(self):
        # 返回输入框内容
        self.savepic("../ok/")
        self.im=PIL.Image.open(self.dataset[self.index])
        self.img=ImageTk.PhotoImage(self.im)
        self.imLabel.configure(image=self.img)
       
    def error_fn(self):
        self.savepic("../fail/")
        self.im=PIL.Image.open(self.dataset[self.index])
        self.img=ImageTk.PhotoImage(self.im)
        self.imLabel.configure(image=self.img)
    def none_fn(self):
        self.savepic("../nores/")
        self.im=PIL.Image.open(self.dataset[self.index])
        self.img=ImageTk.PhotoImage(self.im)
        self.imLabel.configure(image=self.img)
    def savepic(self,path):
        iname = self.dataset[self.index].rsplit('/', 1)[-1]
        img = cv2.imread(self.dataset[self.index])
        file = path + iname
        cv2.imwrite(file,img)
        self.index = self.index + 1

    

if __name__ == '__main__':
    GetCode()










# path = '../result/'
# savepath = '../result/'
# dataset = []
# traverse(path,dataset)
# image_size = len(dataset)
# for index in range(0,image_size):
#     print(dataset[index])
#     img = cv2.imread(dataset[index])
#     cv2.imshow("1",img)
#     cv2.waitKey(0)
#     s = input("Enter your input:")
#     iname = dataset[index].rsplit('/', 1)[-1]
#     if s == "1":
#         file = "../ok" + iname
#         cv2.imwrite(file,img)
#     if s == "2":
#         file = "../fail" + iname
#         cv2.imwrite(file,img)
#     if s == "3":
#         file = "../nores" + iname
#         cv2.imwrite(file,img)

#     #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    