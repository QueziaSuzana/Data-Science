import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from PIL import Image
import customtkinter as CTk
from tkinter import ttk
import math as m
import pandas as pd
import tensorflow as tf
import pickle as pk

screen_height = 350
# Gerar imagem em tons de cinza
def showGrayScale(image_path):
   img = np.asarray(Image.open(image_path).convert('L'))
   plt.title('Imagem em tons de cinza')
   plt.imshow(img, cmap='gray', vmin=0, vmax=255)
   plt.colorbar()
   plt.show()

def showGrayScaleHistogramPage(image_path, master):
   img = cv2.imread(image_path, 0) 
   
   newWindow = CTk.CTkToplevel(master)
   newWindow.title("Histograma de tons de cinza") 
   newWindow.geometry("200x100")
   frm_0 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_0.pack()
   btn_0 = CTk.CTkButton(frm_0, text='Histograma 256 tons de cinza', command= lambda:showGrayScaleHistogram256(img))
   btn_0.pack(side=tk.LEFT, pady=10)

   frm_1 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_1.pack()
   btn_1 = CTk.CTkButton(frm_1, text='Histograma 16 tons de cinza', command= lambda:showGrayScaleHistogram16(img))
   btn_1.pack(side=tk.LEFT, pady=10)
   
def showGrayScaleHistogram256(img):
   # create the histogram
   histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
   plt.title('Histograma com 256 tons de cinza')
   plt.xlabel('Tom')
   plt.ylabel('Pixels')
   tons = list(range(0,256))
   plt.bar(tons,histogram, color='g')
   plt.show()

def showGrayScaleHistogram16(img):
   # create the histogram
   histogram, bin_edges = np.histogram(img, bins=16, range=(0, 256))
   plt.title('Histograma com 16 tons de cinza')
   plt.xlabel('Tom')
   plt.ylabel('Pixels')
   print(histogram)
   tons = list(range(0,16))
   plt.bar(tons, histogram, color='g')
   plt.show()

def showGrayScaleHistogramLines(image_path):
   img = cv2.imread(image_path, 0) 
   # create the histogram
   histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
   # configure and draw the histogram figure
   fig, ax = plt.subplots()
   ax.set_title("Histograma de tons de cinza")
   ax.set_xlabel("Tom")
   ax.set_ylabel("Pixels")
   ax.set_xlim([0, 256])
   ax.plot(bin_edges[0:-1], histogram)  
   plt.show()


def hue(h):
   h_histogram, bin_edges = np.histogram(h, bins=256, range=(0, 256))
   plt.title('Hue (Matiz)')
   plt.xlabel('Valor')
   plt.ylabel('Pixels')
   tons = list(range(0,256))
   plt.bar(tons,h_histogram, color='r')
   plt.show()


def saturation(s):
   s_histogram, bin_edges = np.histogram(s, bins=256, range=(0, 256))
   plt.title('Saturation (Saturação)')
   plt.xlabel('Valor')
   plt.ylabel('Pixels')
   tons = list(range(0,256))
   plt.bar(tons,s_histogram, color='r')
   plt.show()

def value(v):
   v_histogram, bin_edges = np.histogram(v, bins=256, range=(0, 256))
   plt.title('Value (Value)')
   plt.xlabel('Valor')
   plt.ylabel('Pixels')
   tons = list(range(0,256))
   plt.bar(tons,v_histogram, color='r')
   plt.show()

def twoDHistogram (h, v):
   x = h.reshape(-1, 1)
   y = v.reshape(-1, 1)
   x = np.asarray(x)[:, 0]
   y = np.asarray(y)[:, 0]
   # histogram, bin_h, bin_s = np.histogram2d(x, y, bins = (16,8))
   plt.figure(figsize=(10,6))
   hist, xbins, ybins, im = plt.hist2d(x, y, bins=(16, 8))
   plt.title('Histograma 2d H e V(16x8)')
   plt.xlabel('Hue')
   plt.ylabel('Value')
   for i in range(len(ybins)-1):
    for j in range(len(xbins)-1):
        plt.text(xbins[j]+0.5,ybins[i]+0.5, hist.T[i,j], 
                color="w")
   plt.show()

def showHSVColorSpace(master, image_path):
   img = cv2.imread(image_path) 
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   h = hsv[:,:,0]
   s = hsv[:,:,1]
   v = hsv[:,:,2]
   # Open new window
   newWindow = CTk.CTkToplevel(master)
   newWindow.title("Histograma HSV") 
   newWindow.geometry("200x200")

   frm_0 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_0.pack()
   btn_0 = CTk.CTkButton(frm_0, text='Hue', command= lambda:hue(h))
   btn_0.pack(side=tk.LEFT, pady=10)

   frm_1 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_1.pack()
   btn_1 = CTk.CTkButton(frm_1, text='Saturation', command= lambda:saturation(s))
   btn_1.pack(side=tk.LEFT, pady=10)

   frm_2 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_2.pack()
   btn_2 = CTk.CTkButton(frm_2, text= 'Value', command= lambda:value(v))
   btn_2.pack(side=tk.LEFT, pady=10)
   
   frm_3 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_3.pack()
   btn_2 = CTk.CTkButton(frm_3, text= 'Histograma 2d H e V(16x8)', command= lambda:twoDHistogram(h, v))
   btn_2.pack(side=tk.LEFT, pady=10)

def showImagePlot(image_path):
   img = np.asarray(Image.open(image_path))
   plt.title('Imagem original')
   plt.imshow(img)
   plt.colorbar()
   plt.show()

# Arthur methods
def index(l, value) -> int:
   for i in range(len(l)):
      if l[i] == value:
         return i
   return -1
def generate_grayscale_image_from(original_image, shades: int = 16) -> np.ndarray:
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    result = 255 * np.floor(gray * shades + 0.5) / (shades - 1)
    return result.clip(0, 255).astype(np.uint8)

def operation_coocurrence(image) -> "list[np.ndarray]":
    tmp_image: np.ndarray = generate_grayscale_image_from(image)
    uniques = np.sort(np.unique(tmp_image))

    result1 = np.zeros(shape=(len(uniques), len(uniques)))
    result2 = np.zeros(shape=(len(uniques), len(uniques)))
    result4 = np.zeros(shape=(len(uniques), len(uniques)))
    result8 = np.zeros(shape=(len(uniques), len(uniques)))
    result16 = np.zeros(shape=(len(uniques), len(uniques)))
    result32 = np.zeros(shape=(len(uniques), len(uniques)))
    result = [result1, result2, result4, result8, result16, result32]

    for i in range(tmp_image.shape[0]):
        for j in range(tmp_image.shape[1]):
            for c, k in enumerate(result):
                if i + m.pow(2, c) < tmp_image.shape[0] and j + m.pow(2, c) < tmp_image.shape[1]:
                    k[index(uniques, tmp_image[i, j])] \
                        [index(uniques, tmp_image[i + int(m.pow(2, c)), j + int(m.pow(2, c))])] += 1

    for i in result:
        i = i / sum(i)
    return result
 
def entropy(m: np.ndarray) -> float:
    result = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] > 0:
                result += m[i][j] * (np.log2(m[i][j]))
    return abs(result)
 
def homogeneity(m) -> float:
    result = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            result += m[i][j] / (1 + abs(i - j))
    return result


def contrast(m) -> float:
    result = 0
    for i in range(len(m)):
        for j in range(len(m[0])):
            result += m[i][j] * (i - j) ** 2
    return result
 
def showHaralickTable (image_path, master):
   image = cv2.imread(image_path)
   result = operation_coocurrence(image)
   
   # Open new window
   newWindow = CTk.CTkToplevel(master)
   newWindow.title("Descritores de Haralick") 
   newWindow.geometry("600x175")
   # treeview 
   table = ttk.Treeview(newWindow, columns = ('entropia', 'homogeneidade', 'contraste'), show = 'headings')
   table.heading('entropia', text = 'Entropia')
   table.column('entropia', anchor=tk.CENTER)
   table.heading('homogeneidade', text = 'Homogeneidade')
   table.column('homogeneidade', anchor=tk.CENTER)
   table.heading('contraste', text = 'Contraste')
   table.column('contraste', anchor=tk.CENTER)
   table.pack(fill = 'both', expand = True)
   
   for i in range(6):
      data = (round(entropy(result[i]), 5), round(homogeneity(result[i]),5), round(contrast(result[i])),5)
      table.insert(parent = '', index = 0, values = data)

def operation_hue_invariants(image) -> tuple:
    result = generate_grayscale_image_from(image, shades=256)
    result_grayscale = cv2.HuMoments(cv2.moments(result, binaryImage=False)).flatten()
    
    tmp_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    result_h = cv2.HuMoments(cv2.moments(tmp_image[:, :, 0], binaryImage=False)).flatten()
    result_s = cv2.HuMoments(cv2.moments(tmp_image[:, :, 1], binaryImage=False)).flatten()
    result_v = cv2.HuMoments(cv2.moments(tmp_image[:, :, 2], binaryImage=False)).flatten()

    return (result_grayscale, (result_h, result_s, result_v))
 
def showHuVariants (image_path, master):
   image = cv2.imread(image_path)
   result = operation_hue_invariants(image)
   
   # Open new window
   newWindow = CTk.CTkToplevel(master)
   newWindow.title("Momentos invariantes de Hu") 
   newWindow.geometry("920x125")
   # treeview 
   table = ttk.Treeview(newWindow, columns = ('H[0]', 'H[1]', 'H[2]', 'H[3]', 'H[4]', 'H[5]', 'H[6]'), show = 'headings')
   table.heading('H[0]', text = 'H[0]')
   table.column('H[0]', anchor=tk.CENTER, width=130)
   table.heading('H[1]', text = 'H[1]')
   table.column('H[1]', anchor=tk.CENTER, width=130)
   table.heading('H[2]', text = 'H[2]')
   table.column('H[2]', anchor=tk.CENTER, width=130)
   table.heading('H[3]', text = 'H[3]')
   table.column('H[3]', anchor=tk.CENTER, width=130)
   table.heading('H[4]', text = 'H[4]')
   table.column('H[4]', anchor=tk.CENTER, width=130)
   table.heading('H[5]', text = 'H[5]')
   table.column('H[5]', anchor=tk.CENTER, width=130)
   table.heading('H[6]', text = 'H[6]')
   table.column('H[6]', anchor=tk.CENTER, width=130)
   table.pack(fill = 'both', expand = True)
   
   # print('HuInvariants GrayScale256: ', result[0])
   # print(result[1][0])
   # print(result[1][1])
   # print(result[1][2])
   
   # GrayScale256
   data = (result[0][0], result[0][1], result[0][2], result[0][3], result[0][4], result[0][5], result[0][6])
   table.insert(parent = '', index = 0, values = data)
   # HSV
   for i in range(3):
      data = (result[1][i][0], result[1][i][1], result[1][i][2], result[1][i][3], result[1][i][4], result[1][i][5], result[1][i][6])
      table.insert(parent = '', index = i+1, values = data)

def predictSVMBin(image, newWindow):
   result = operation_coocurrence(image)
   data = []
   for i in range(6):
      data += (entropy(result[i]), homogeneity(result[i]), contrast(result[i]))
   data = np.array(data)
   print(data)
   print(data.shape)
   df = pd.DataFrame(data=[data], columns=["entropy(1,1)","homogeneity(1,1)","contrast(1,1)","entropy(2,2)","homogeneity(2,2)","contrast(2,2)","entropy(4,4)","homogeneity(4,4)","contrast(4,4)","entropy(8,8)","homogeneity(8,8)","contrast(8,8)","entropy(16,16)","homogeneity(16,16)","contrast(16,16)","entropy(32,32)", "homogeneity(32,32)", "contrast(32,32)"])
   model = pk.load(open('best_model_svm_binary.pkl', 'rb'))
   predictions = model.predict(df)  # class probabilities

   predicted_classes = (predictions >= 0.5).astype(int)  # binary only
   print(predicted_classes)
   if 0 in predicted_classes: 
      resultado = 'Não é celula cancerígena'
   else:
      resultado = 'É celula cancerígena'
   print(resultado)  
   global screen_height 
   screen_height += 30
   geo = '350x'+str(screen_height)
   print(geo)
   newWindow.geometry(geo) 
   frm_5 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_5.pack()
   resultadoClass = CTk.CTkLabel(frm_5, text= 'SVM Binário - ' + resultado,  justify=tk.LEFT, font=('Helvetica 18 bold', 18))
   resultadoClass.pack(side=tk.LEFT, pady=10)

def predictSVMMulti(image, newWindow):
   result = operation_coocurrence(image)
   data = []
   for i in range(6):
      data += (entropy(result[i]), homogeneity(result[i]), contrast(result[i]))
   data = np.array(data)
   df = pd.DataFrame(data=[data], columns=["entropy(1,1)","homogeneity(1,1)","contrast(1,1)","entropy(2,2)","homogeneity(2,2)","contrast(2,2)","entropy(4,4)","homogeneity(4,4)","contrast(4,4)","entropy(8,8)","homogeneity(8,8)","contrast(8,8)","entropy(16,16)","homogeneity(16,16)","contrast(16,16)","entropy(32,32)", "homogeneity(32,32)", "contrast(32,32)"])
   model = pk.load(open('best_model_svm_multiclass.pkl', 'rb'))
   predictions = model.predict(df)  # class probabilities
   print(predictions)
   if 0 in predictions: 
      resultado = 'Não é celula cancerígena'
   elif 1 in predictions:
      resultado = 'ASC-H'
   elif 2 in predictions:
      resultado = 'ASC-US'
   elif 3 in predictions:
      resultado = 'HSIL'
   elif 4 in predictions:
      resultado = 'LSIL'
   elif 5 in predictions:
      resultado = 'SCC'
      
   global screen_height 
   screen_height += 30
   geo = '350x'+str(screen_height)
   newWindow.geometry(geo) 
   frm_5 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_5.pack()
   resultadoClass = CTk.CTkLabel(frm_5, text= 'SVM Multiclasse - ' + resultado,  justify=tk.LEFT, font=('Helvetica 18 bold', 18))
   resultadoClass.pack(side=tk.LEFT, pady=10)
   
def predictEFCBin(image, newWindow):
   model = pk.load(open('model_efn_binary_8_3.pkl', 'rb'))
   X_data = []
   X_data.append(image)

   X_data = np.array(X_data)

   predictions = model.predict(X_data)  # class probabilities
   predicted_classes = (predictions >= 0.5).astype(int)  # binary only
   print(predicted_classes)
   if 0 in predicted_classes: 
      resultado = 'Não é celula cancerígena'
   else:
      resultado = 'É celula cancerígena'
   
   global screen_height 
   screen_height += 30
   geo = '350x'+str(screen_height)
   newWindow.geometry(geo) 
   frm_5 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_5.pack()
   resultadoClass = CTk.CTkLabel(frm_5, text= 'Efc Binário - ' + resultado,  justify=tk.LEFT, font=('Helvetica 18 bold', 18))
   resultadoClass.pack(side=tk.LEFT, pady=10)

def predictEFCMulti(image, newWindow):
   image = cv2.resize(image, (224, 224))
   model = tf.keras.models.load_model("model_efn_multiclass_512_2.h5")
   X_data = []
   X_data.append(image)

   X_data = np.array(X_data)

   predictions = model.predict(X_data)  # class probabilities
   predicted_classes = predictions.argmax(axis=1)  # multiclass only
   print(predicted_classes)
   if 0 in predicted_classes: 
      resultado = 'Não é celula cancerígena'
   elif 1 in predicted_classes:
      resultado = 'ASC-H'
   elif 2 in predicted_classes:
      resultado = 'ASC-US'
   elif 3 in predicted_classes:
      resultado = 'HSIL'
   elif 4 in predicted_classes:
      resultado = 'LSIL'
   elif 5 in predicted_classes:
      resultado = 'SCC'
   print(resultado)
   global screen_height 
   screen_height += 30
   geo = '350x'+str(screen_height)
   newWindow.geometry(geo) 
   frm_5 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_5.pack()
   resultadoClass = CTk.CTkLabel(frm_5, text= 'Efc Multiclasse - ' + resultado,  justify=tk.LEFT, font=('Helvetica 18 bold', 18))
   resultadoClass.pack(side=tk.LEFT, pady=10)
def showClassifications(master, image_path):
   # Open new window
   global screen_height 
   screen_height = 350
   newWindow = CTk.CTkToplevel(master)
   newWindow.title("Classificação") 
   newWindow.geometry("350x300")
   print(image_path)
   image = cv2.imread(image_path)
   
   frm_0 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_0.pack()
   btn_0 = CTk.CTkButton(frm_0, text='SVM Binário', command= lambda:predictSVMBin(image, newWindow))
   btn_0.pack(side=tk.LEFT, pady=10)

   frm_1 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_1.pack()
   btn_1 = CTk.CTkButton(frm_1, text='SVM Multiclasse (6 classes)', command= lambda:predictSVMMulti(image, newWindow))
   btn_1.pack(side=tk.LEFT, pady=10)
   
   frm_2 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_2.pack()
   btn_2 = CTk.CTkButton(frm_2, text= 'EfficientNet Binário', command= lambda:predictEFCBin(image, newWindow))
   btn_2.pack(side=tk.LEFT, pady=10)
   
   frm_3 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_3.pack()
   btn_2 = CTk.CTkButton(frm_3, text= 'EfficientNet Multiclasse (6 classes)', command= lambda:predictEFCMulti(image, newWindow))
   btn_2.pack(side=tk.LEFT, pady=10)
   
   frm_4 = CTk.CTkFrame(newWindow, fg_color='#242424')
   frm_4.pack()
   resultadoL = CTk.CTkLabel(frm_4, text='Resultado:',  justify=tk.LEFT, font=('Helvetica 18 bold', 20))
   resultadoL.pack(side=tk.LEFT, pady=10)
   
   

