#IMPORTS:
import tkinter as tk
import os
import cv2
import numpy as np
from deepface import DeepFace

#Criar dataset, antes de tudo:
dir = "Dataset"
os.makedirs(dir, exist_ok=True)



#FUNCOES:
def criarDataset(nome):
    usuario = os.path.join(dir, nome)
    os.makedirs(usuario, exist_ok=True)

    captura = cv2.VideoCapture(0)
    qtdCapturas = 100
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while qtdCapturas > 0:
        ret, frame = captura.read()
        if not ret:
            print("Nao foi possivel capturar imagem.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in rostos:
            img_rosto = frame[y:y+h, x:x+w]
            caminho_rosto = os.path.join(usuario, f"{nome}_{100 - qtdCapturas + 1}.jpg")
            cv2.imwrite(caminho_rosto, img_rosto)
            qtdCapturas -= 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Exibe quantas imagens faltam
            cv2.putText(frame, f"Faltam: {qtdCapturas}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break  # salva só uma face por frame

        cv2.imshow("Captura de face pela camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Captura interrompida pelo usuario.")
            break

    captura.release()
    cv2.destroyAllWindows()
    print("Captura finalizada.")

def treinarDataset():
    agrupamento = {}
    for i in os.listdir(dir):
        usuario = os.path.join(dir, i)
        if os.path.isdir(usuario):
            agrupamento[i] = []
            for emotion_folder in os.listdir(usuario):
                emotion_dir = os.path.join(usuario, emotion_folder)
                if os.path.isdir(emotion_dir):
                    for img_nome in os.listdir(emotion_dir):
                        img_caminho = os.path.join(emotion_dir, img_nome)

                        try:
                            embedding = DeepFace.represent(img_caminho, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                            agrupamento[i].append(embedding)
                        except Exception as e:
                            print("Erro ao treinar imagens:", e)
    np.save("embedding.npy", agrupamento)
    print("Treinamento finalizado.")
    return agrupamento

def mostrarEmocao(agrupamentos):
    captura = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    while True:
        ret, frame = captura.read()
        if not ret:
            print("Falha na captura de imagem")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in rostos:
            img_rosto = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            try:
                analysis = DeepFace.analyze(img_rosto, actions=["emotion"], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]

                emotion = max(analysis["emotion"], key=analysis["emotion"].get)

                face_embedding = DeepFace.represent(img_rosto, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                match = None
                max_similarity = -1
                for pessoa, embeddings in agrupamentos.items():
                    for embed in embeddings:
                        similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = pessoa

                if max_similarity > 0.7:
                    label = f"{match} ({max_similarity:.2f})"
                else:
                    label = "Pessoa nao reconhecida"

                display_text = f"Nome: {label} | Emocao: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except Exception as e:
                print("Nao foi possivel reconhecer rosto: ", e)

        cv2.imshow("Reconhecimento facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()

# Deixar automático o treinamento do dataset
def criarMaisTreinarDataset(nome_usuario):
    criarDataset(nome_usuario)
    treinarDataset()

# Só permite o reconhecimento facial caso o dataset esteja treinado
def reconhecimentoEmocao():
    if os.path.exists("embedding.npy"):
        agrupamentos = np.load("embedding.npy", allow_pickle=True).item()
        mostrarEmocao(agrupamentos)
    else:
        print("É necessário criar um dataset antes de realizar o reconhecimento!")


#CODIGO DA UI:
# Tela Inicial:
janela = tk.Tk()
janela.title("Identificador de Emoções")
janela.geometry("1024x768")
janela.configure(bg="#3e4e60")

# Título:
titulo = tk.Label(janela, text="TCC", font=("Arial", 36))
titulo.pack(pady=10)
titulo.configure(bg="#3e4e60", fg="white")

# Nome do Usuário:
nome = tk.Entry(janela, font=("Arial", 20))
nome.pack(pady=10)
nome.configure(bg="#667f98", fg="white")

# Botão Criar Dataset:
botaoDataset = tk.Button(janela, font=("Arial", 16), text="Criar Dataset", command=lambda: criarMaisTreinarDataset(nome.get()))
botaoDataset.pack(pady=10)
botaoDataset.configure(bg="#667f98", fg="white")

# Botão Mostrar Emoções:
botaoEmocao = tk.Button(janela, font=("Arial", 16), text="Iniciar Reconhecimento", command=reconhecimentoEmocao)
botaoEmocao.pack()
botaoEmocao.configure(bg="#667f98", fg="white")

janela.mainloop()
