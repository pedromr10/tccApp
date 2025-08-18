#IMPORTS:
import os

#MAIN CODE:

#Cria uma pasta para colocar os arquivos de print da tela:
#OBS: Atualmente servirá apenas para colocar imagens para teste ainda nao vai fazer a captura.
'''
dir = "Gameplay"
os.makedirs(dir, exist_ok=True)
'''

from ultralytics import YOLO

#carrega o modelo pre treinado do yolo
model = YOLO('yolov8x.pt')  # versão pequena, rápida

#faz a detecao a partir de uma imagem (por enquanto hehe)
results = model('bf.png')

#mostra a primeira resolucao
results[0].show()  
