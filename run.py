#FOLLOWED TUTORIAL: Tech With Raza (https://www.youtube.com/@TechWithAhmedRaza)

#imports:
import os
import cv2
import numpy as np
from deepface import DeepFace

#Criar dataset:
dir = "Dataset"
os.makedirs(dir, exist_ok=True)

def criarDataset(nome):
    usuario = os.path.join(dir, nome)
    os.makedirs(usuario, exist_ok=True)

    #Captura da imagem pelo cv2:
    captura = cv2.VideoCapture(0)
    qtdCapturas = 100
    while True:
        ret, frame = captura.read()
        #ret funciona como um booleano, True quando a captura da imagem 
        #funcionou corretamente e false quando ocorre o oposto.
        if not ret:
            print("Nao foi possivel capturar imagem.")
            break

        #deixa a imagem cinza para fazer a captura, deve facilitar (pesquisarei sobre)
        gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
        rostos = 






        #OBS: cascade tirado de: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
