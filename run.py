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

        #Aparentemente o openCv detecta faces transformando imagens coloridas em tons de cinza e depois usa o classficador haardcascade.
        gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
        rostos = cv2.CascadeClassifier(cv2.data.haardcascades+"haardcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for(x, y, w, h) in rostos:
            qtdCapturas-=1
            img_rosto = frame[y:y+h, x:x+w]
            caminho_rosto = os.path.join(usuario, f"{nome}_{qtdCapturas}.jpg")
            cv2.imwrite(caminho_rosto, img_rosto)

            cv2.rectangle((frame, (x,y), (x+w, y+h), (255, 0, 0)), 2)
            cv2.imshow("Captura de face pela camera")

            if cv2.waitKey(1) & 0xFF == ord('q') or qtdCapturas >= 100:
                break
    captura.release()
    cv2.destroyAllWindows()

#TREINAMENTO DE DATASET:
def treinar_dataset():
    agrupamento={}
    for i in os.listdir(dir):
        usuario = os.path.join(dir, i)
        if os.path.isdir(usuario):
            agrupamento[i] = []
            for img_nome in  os.listdir(usuario):
                img_caminho = os.path.join(usuario, img_nome)
                try:
                    agrupamento = DeepFace.represent(img_caminho, model_name = "Facenet", enforce_detection = False)[0]["agrupamento"]
                    agrupamento[i].append(agrupamento)
                except Exception as e:
                    print("Erro ao treinar imagens")
    return agrupamento




#OBS: cascade tirado de: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml