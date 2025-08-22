#IMPORTS:
import os, cv2
from ultralytics import YOLO


#MAIN CODE:

#Cria uma pasta para colocar os arquivos de print da tela:
#OBS: Atualmente servirá apenas para colocar imagens para teste ainda nao vai fazer a captura.

datasets_emocoes = "Dataset"

pasta_gameplays = "Gameplay"
os.makedirs(pasta_gameplays, exist_ok=True)

#carrega o modelo pre treinado do yolo
model = YOLO('yolov8x.pt')  # versão pequena, rápida

# Pega as imagens de dentro das subpastas do Dataset
imagens = []
for root, dirs, files in os.walk(datasets_emocoes):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            imagens.append(os.path.normpath(os.path.join(root, f)))

imagens.sort()

for i, img in enumerate(imagens):
    caminho = img
    frame = cv2.imread(caminho)

    if frame is None:
        print(f"Falha ao abrir {caminho}")
        continue

    resultados = model(frame)

    frame_pronto = resultados[0].plot()

    nome_arquivo = os.path.basename(caminho)
    saida = os.path.join(pasta_gameplays, nome_arquivo)

    cv2.imwrite(saida, frame_pronto)

    print(f"Salvo: {saida}")