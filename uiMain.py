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
    nome_usuario = nome.strip().lower().replace(" ", "_")
    usuario_dir = os.path.join(dir, nome_usuario)

    nome_usuario_cont = 1
    while os.path.exists(usuario_dir):
        usuario_dir = os.path.join(dir, f"{nome_usuario}_{nome_usuario_cont}")
        nome_usuario_cont += 1

    os.makedirs(usuario_dir, exist_ok=True)

    captura = cv2.VideoCapture(0)
    qtdCapturas = 100
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cont_emocoes = {}

    while qtdCapturas > 0:
        ret, frame = captura.read()
        if not ret:
            print("Nao foi possivel capturar imagem.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rostos = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in rostos:
            img_rosto = frame[y:y+h, x:x+w]

            try:
                result = DeepFace.analyze(img_rosto, actions=['emotion'], enforce_detection=False)
                emotion = result[0]["dominant_emotion"] if isinstance(result, list) else result["dominant_emotion"]

                cont_emocoes.setdefault(emotion, 0)
                if cont_emocoes[emotion] < 100:
                    cont_emocoes[emotion] += 1
                    emotion_dir = os.path.join(usuario_dir, emotion)
                    os.makedirs(emotion_dir, exist_ok=True)
                    nome_arquivo = f"{emotion}_{cont_emocoes[emotion]}.jpg"
                    caminho_arquivo = os.path.join(emotion_dir, nome_arquivo)
                    cv2.imwrite(caminho_arquivo, img_rosto)

                    # Salva a emoção num .txt
                    with open(os.path.join(usuario_dir, "emotions.txt"), "a", encoding="utf-8") as f:
                        f.write(f"{emotion}\n")

                    qtdCapturas -= 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{emotion} | Faltam: {qtdCapturas}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print("Erro ao analisar emoção:", e)

            break  # captura só um rosto por frame

        cv2.imshow("Captura de face e emoção", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Captura interrompida.")
            break
        
        if qtdCapturas <= 0:
            print("Gostaria de continuar capturando imagens?")
            continuar_captura = int(input("1. SIM | 2. NAO "))
            if continuar_captura == 1:
                qtdCapturas = 100

    captura.release()
    cv2.destroyAllWindows()
    print("Captura finalizada.")

def mostrarEmocao():
    print("teste")



#CODIGO DA UI:
#Tela inicial:
janela = tk.Tk()
janela.title("Identificador de emocoes")
janela.geometry("600x300")
janela.configure(bg = "lightblue")

#Titulo:
titulo = tk.Label(janela, text="TCC", font=("Arial", 14))
titulo.pack(pady=10)
titulo.configure(bg = "lightblue", fg = "white")

#NomeUsuario:
nome = tk.Entry(janela)
nome.pack(pady = 10)
nome.configure(bg = "darkblue", fg = "white")

#BotãoCriarDataset
botaoDataset = tk.Button(janela, text="Criar Dataset", command=lambda: criarDataset(nome.get()))
botaoDataset.pack(pady=10)
botaoDataset.configure(bg = "darkblue", fg = "white")

#a parte 2 do codigo poderia ser automatica.

#BotãoMostrarEmocoes
botaoEmocao = tk.Button(janela, text="Emocoes", command=mostrarEmocao)
botaoEmocao.pack()
botaoEmocao.configure(bg = "darkblue", fg = "white")

# Inicia o loop da interface
janela.mainloop()
