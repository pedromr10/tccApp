# IMPORTS:
import tkinter as tk
from tkinter import ttk
import threading
import os
import cv2
import numpy as np
from deepface import DeepFace

# Criar dataset, antes de tudo:
dir = "Dataset"
os.makedirs(dir, exist_ok=True)

# FUNCOES:
def detectarCamera():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ret, frame = cap.read()
        if ret:
            return cap
        cap.release()
    print("Nenhuma câmera encontrada")
    return None

def criarDataset(nome):

    def iniciar_captura(qtd):
        nomeBase = nome.strip().lower().replace(" ", "_")

        contNome = 1
        usuario = os.path.join(dir, f"{nomeBase}_{contNome}")
        while os.path.exists(usuario):
            contNome += 1
            usuario = os.path.join(dir, f"{nomeBase}_{contNome}")

        os.makedirs(usuario, exist_ok=True)

        captura = detectarCamera()
        if captura is None:
            print("Nenhuma câmera disponível")
            return

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        qtdCapturas = qtd

        while qtdCapturas > 0:
            ret, frame = captura.read()
            if not ret:
                print("Nao foi possivel capturar imagem.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rostos = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in rostos:
                img_rosto = frame[y:y+h, x:x+w]
                caminho_rosto = os.path.join(usuario, f"{nomeBase}_{contNome}_{qtd - qtdCapturas + 1}.jpg")
                cv2.imwrite(caminho_rosto, img_rosto)
                qtdCapturas -= 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Faltam: {qtdCapturas}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

            cv2.imshow("Captura de face pela camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Captura interrompida pelo usuario.")
                break

        captura.release()
        cv2.destroyAllWindows()
        print("Captura finalizada.")

    def iniciar_captura_de_verdade(janelaBarra=None):
        try:
            qtd = int(qtdInserida.get())
        except ValueError:
            print("Valor inválido")
            if janelaBarra:
                janelaBarra.destroy()
            return

        janelaQtdCapturas.destroy()
        iniciar_captura(qtd)

    janelaQtdCapturas = tk.Toplevel()
    janelaQtdCapturas.title("Quantidade de capturas desejadas")
    janelaQtdCapturas.geometry("600x400")
    janelaQtdCapturas.configure(bg="#3e4e60")

    tk.Label(janelaQtdCapturas, text="Quantidade de capturas: ", font=("Arial", 36),
             bg="#3e4e60", fg="white").pack(pady=10)

    qtdInserida = tk.Entry(janelaQtdCapturas, font=("Arial", 36), bg="#3e4e60", fg="white")
    qtdInserida.pack(pady=10)

    tk.Button(janelaQtdCapturas, text="Iniciar", font=("Arial", 36),
              command=lambda: barraDeCarregamento(iniciar_captura_de_verdade),
              bg="#3e4e60", fg="white").pack(pady=10)

def treinarDataset(janelaBarra=None):

    agrupamento = {}
    for i in os.listdir(dir):
        usuario = os.path.join(dir, i)
        if os.path.isdir(usuario):
            agrupamento[i] = []
            for img_nome in os.listdir(usuario):
                img_caminho = os.path.join(usuario, img_nome)

                try:
                    embedding = DeepFace.represent(img_caminho, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    agrupamento[i].append(embedding)
                except Exception as e:
                    print("Erro ao treinar imagem:", e)

    np.save("embedding.npy", agrupamento)
    print("Treinamento finalizado.")
    return agrupamento

def mostrarEmocao(agrupamentos, janelaBarra=None):

    if janelaBarra:
        janelaBarra.destroy()

    captura = detectarCamera()
    if captura is None:
        print("Nenhuma câmera disponível")
        return

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
                print("Emocao detectada: " + emotion)

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

def reconhecimentoEmocao(janelaBarra=None):
    if os.path.exists("embedding.npy"):
        agrupamentos = np.load("embedding.npy", allow_pickle=True).item()
        mostrarEmocao(agrupamentos, janelaBarra)
    else:
        print("É necessário criar um dataset antes de realizar o reconhecimento!")

# Barra de carregamento
def barraDeCarregamento(func, *args):
    janelaBarraDeCarregamento = tk.Toplevel()
    janelaBarraDeCarregamento.title("Carregando...")
    janelaBarraDeCarregamento.geometry("600x100")
    janelaBarraDeCarregamento.configure(bg="#3e4e60")

    tk.Label(janelaBarraDeCarregamento, text="Por favor, aguarde alguns instantes...",
             font=("Arial", 20), bg="#3e4e60", fg="white").pack(pady=10)

    barra = ttk.Progressbar(janelaBarraDeCarregamento, orient="horizontal",
                            length=300, mode="indeterminate")
    barra.pack(pady=5)
    barra.start()

    def tarefa():
        try:
            func(*args)
        finally:
            barra.stop()
            janelaBarraDeCarregamento.destroy()

    threading.Thread(target=tarefa).start()

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

#Botao Criar Dataset:
botaoDataset = tk.Button(janela, font=("Arial", 16), text="Criar Dataset", command=lambda: criarDataset(nome.get()))
botaoDataset.pack(pady=10)
botaoDataset.configure(bg="#667f98", fg="white")

#botao para treinar o dataset para reconhecimento de perfil (pro deepface saber que vc é vc!!!):
botaoTreinamento = tk.Button(janela, font=("Arial", 16), text= "Treinar dataset", command=lambda: barraDeCarregamento(treinarDataset))
botaoTreinamento.pack()
botaoTreinamento.configure(bg="#667f98", fg="white")

# Botão Mostrar Emoções:
botaoEmocao = tk.Button(janela, font=("Arial", 16), text="Iniciar Reconhecimento", command=lambda: barraDeCarregamento(reconhecimentoEmocao))
botaoEmocao.pack()
botaoEmocao.configure(bg="#667f98", fg="white")

janela.mainloop()
