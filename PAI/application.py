import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from tkinter import Scrollbar
from PIL import Image, ImageTk
from image_manipulations import * 
import customtkinter as CTk

class Application:
    def appDefinitions(self, master=None):     
        # setting title and basic size to our App
        self.title = 'Processamento e análise de imagens'
        self.master = master
        master.title(self.title)
        self.geometry = master.geometry('750x610')

        # Setting buttons
        self.frm_0 = CTk.CTkFrame(master, fg_color='#242424')
        self.frm_0.pack()
        self.uploadButton(self.frm_0)

        # Image frame
        self.frm_1 = CTk.CTkFrame(master)
        self.frm_1.pack()
        self.defaultImageLabel = CTk.CTkLabel(self.frm_1, text='Nenhuma imagem selecionada')
        self.defaultImageLabel.pack( pady= 10)
        self.showImageButtons()

    def setTitle(self,title):
        self.title = self.master.title(title)
    
    def showImageButtons(self):
        # Button visualizar tons de cinza
        self.frm_00 = CTk.CTkFrame(self.master, fg_color='#242424')
        self.frm_00.pack(padx=5)
        self.btn_00 = CTk.CTkButton(self.frm_00, text ='Visualizar imagem', command= lambda:showImagePlot(self.image_path))
        self.btn_00.pack(side=tk.LEFT, pady=10, padx=5)

        # Button visualizar tons de cinza
        self.frm_2 = CTk.CTkFrame(self.master, fg_color='#242424')
        self.frm_2.pack()
        self.btn_0 = CTk.CTkButton(self.frm_2, text= 'Visualizar tons de cinza', command= lambda:showGrayScale(self.image_path))
        self.btn_0.pack(side=tk.LEFT, pady=10)
        #self.btn_0.grid(column = 2, row=3)

        # Button Gerar histograma tons de cinza
        self.btn_1 = CTk.CTkButton(self.frm_2, text='Gerar histograma tons de cinza', command= lambda:showGrayScaleHistogramPage(self.image_path, self.master))
        self.btn_1.pack(side=tk.RIGHT, pady=10, padx = 10)

        # Button Gerar histograma HSV
        self.frm_3 = CTk.CTkFrame(self.master, fg_color='#242424')
        self.frm_3.pack()
        self.btn_2 = CTk.CTkButton(self.frm_3, text='Gerar histograma HSV', command= lambda:showHSVColorSpace(self.master, self.image_path))
        self.btn_2.pack(side=tk.LEFT, pady=10)

        # Button visualizar tons de cinza atraves dos descritores de haralick
        self.btn_3 = CTk.CTkButton(self.frm_3, text='Visualizar tons de cinza de Haralick', command= lambda:showHaralickTable(self.image_path, self.master))
        self.btn_3.pack(side=tk.RIGHT, pady=10, padx = 10)

        # Button visualizar caracterizacao de tons de cinza e canais de HU
        self.frm_4 = CTk.CTkFrame(self.master, fg_color='#242424')
        self.frm_4.pack()
        self.btn_4 = CTk.CTkButton(self.frm_4, text='Visualizar caracterização de HU', command= lambda:showHuVariants(self.image_path, self.master))
        self.btn_4.pack(side=tk.LEFT, pady=10)

        # Button de visualizar classificacao
        self.btn_5 = CTk.CTkButton(self.frm_4, text='Visualizar classificação', command= lambda:showClassifications(self.master, self.image_path ))
        self.btn_5.pack(side=tk.RIGHT, pady=10, padx = 10)

        # Button Sair
        self.frm_5 = CTk.CTkFrame(self.master, fg_color='#242424')
        self.frm_5.pack()
        self.btn_6 = CTk.CTkButton(self.frm_5, fg_color= '#FF1E27', text='Sair', command=self.master.destroy, hover_color='#EB1C24')
        self.btn_6.pack(side=tk.LEFT, pady=10)
        self.disableButtons()

    def enableButtons(self):
        self.btn_00.configure(state=tk.NORMAL)
        self.btn_0.configure(state=tk.NORMAL)
        self.btn_1.configure(state=tk.NORMAL)
        self.btn_2.configure(state=tk.NORMAL)
        self.btn_3.configure(state=tk.NORMAL)
        self.btn_4.configure(state=tk.NORMAL)
        self.btn_5.configure(state=tk.NORMAL)
        self.btn_6.configure(state=tk.NORMAL)

    def disableButtons(self):
        self.btn_00.configure(state=tk.DISABLED)
        self.btn_0.configure(state=tk.DISABLED)
        self.btn_1.configure(state=tk.DISABLED)
        self.btn_2.configure(state=tk.DISABLED)
        self.btn_3.configure(state=tk.DISABLED)
        self.btn_4.configure(state=tk.DISABLED)
        self.btn_5.configure(state=tk.DISABLED)

    def imageUploader(self):
        fileTypes = [('Image files', '*.png;*.jpg;*.jpeg')]
        path = tk.filedialog.askopenfilename(filetypes=fileTypes)
    
        # if file is selected
        if len(path):
            self.image_path = path
            self.img = Image.open(path)
            self.pic = ImageTk.PhotoImage(self.img)
            if self.pic.width() == 1280:
                self.pic = self.pic._PhotoImage__photo.subsample(4)
            if self.pic.width() == 100:
                self.pic = self.pic._PhotoImage__photo.zoom(2)
                
            # re-sizing the app window in order to fit picture and buttom
            self.defaultImageLabel.configure(text= path)
            self.image_label = CTk.CTkLabel(self.frm_1, text='')
            self.image_label.pack(pady=10)
            self.image_label.configure(image=self.pic)
            self.image_label.image = self.pic
            self.upload_button.pack_forget()
            self.btn_8 = CTk.CTkButton(self.frm_0, text='Trocar imagem', command=lambda:self.changeImage())
            self.btn_8.pack(side=tk.LEFT, pady=10)
            self.enableButtons()
            
        # if no file is selected, then we are displaying below message
        else:
            self.defaultImageLabel.configure(text = 'Nenhuma imagem selecionada')
            print('Nenhuma imagem foi selecionada, por favor selecione uma.')

    def uploadButton(self, frame):
        # defining upload buttom
        self.upload_button = CTk.CTkButton(frame, text = 'Selecionar imagem',  command= lambda: self.imageUploader())
        self.upload_button.pack(side=tk.LEFT, pady= 10)

    def changeImage(self):
        self.image_label.pack_forget()

        fileTypes = [('Image files', '*.png;*.jpg;*.jpeg')]
        path = tk.filedialog.askopenfilename(filetypes=fileTypes)

        if len(path):
            self.image_path = path
            self.img = Image.open(path)
            self.pic = ImageTk.PhotoImage(self.img)
            if self.pic.width() == 1280:
                self.pic = self.pic._PhotoImage__photo.subsample(4)
            if self.pic.width() == 100:
                self.pic = self.pic._PhotoImage__photo.zoom(2)
            # re-sizing the app window in order to fit picture and buttom
            self.defaultImageLabel.configure(text= path)
            self.image_label = CTk.CTkLabel(self.frm_1, text='')
            self.image_label.pack(pady=10)
            self.image_label.configure(image=self.pic)
            self.image_label.image = self.pic
        # if no file is selected, then we are displaying below message
            self.enableButtons()
        else:
            self.disableButtons()
            self.defaultImageLabel.configure(text = 'Nenhuma imagem selecionada')
            print('Nenhuma imagem foi selecionada, por favor selecione uma.')

    def __init__(self, master=None):
        self.appDefinitions(master)



def main():
    CTk.set_appearance_mode("Dark") 
    root = CTk.CTk()
    Application(root)
    root.mainloop()
