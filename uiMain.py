# IMPORTS:
import tkinter as tk
from tkinter import ttk
import threading
import os, cv2, mss
import numpy as np
from deepface import DeepFace

# Criar dataset, antes de tudo:
dir = "Dataset"
os.makedirs(dir, exist_ok=True)

# Carrega o modelo DNN:
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

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

def detectarFace(frame, conf_threashold=0.8):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    rostos = []

    for i in range (0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threashold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            rostos.append((x1, y1, x2 - x1, y2 - y1))
    return rostos

def criarDataset(nome):
    #alem de tirar as fotos, faz uma copia mais escura e uma mais clara para um melhor treinamento
    #e deteccao posteriormente.
    def iniciar_captura(qtd):
        nomeBase = nome.strip().lower().replace(" ", "_")
        #valores para o data augmentation:
        valor_claro = 50
        valor_escuro = -55
        
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

        qtdCapturas = qtd

        while qtdCapturas > 0:
            ret, frame = captura.read()
            if not ret:
                print("Nao foi possivel capturar imagem.")
                break

            rostos = detectarFace(frame, conf_threashold=0.8)

            for (x, y, w, h) in rostos:
                img_rosto = frame[y:y+h, x:x+w]

                #caminhos para salvamento das imagens dos rostos:
                caminho_rosto = os.path.join(usuario, f"{nomeBase}_{contNome}_{qtd - qtdCapturas + 1}.jpg")
                caminho_rosto_claro = os.path.join(usuario, f"{nomeBase}_{contNome}_{qtd - qtdCapturas + 1}_claro.jpg")
                caminho_rosto_escuro = os.path.join(usuario, f"{nomeBase}_{contNome}_{qtd - qtdCapturas + 1}_escuro.jpg")
                
                #salvamento da imagem original:
                cv2.imwrite(caminho_rosto, img_rosto)

                #alterando as imagens:
                img_clara = cv2.convertScaleAbs(img_rosto, alpha=1, beta=valor_claro)
                img_escura = cv2.convertScaleAbs(img_rosto, alpha=1, beta=valor_escuro)
                #salvamento da imagem mais clara:
                cv2.imwrite(caminho_rosto_claro, img_clara)
                #salvamento da imagem mais escura:
                cv2.imwrite(caminho_rosto_claro, img_escura)

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

    # Capturar tela
    sct = mss.mss()
    monitor = sct.monitors[1]

    while True:

        tela = np.array(sct.grab(monitor))
        tela = cv2.cvtColor(tela, cv2.COLOR_BGRA2BGR)

        ret, frame = captura.read()
        if not ret:
            print("Falha na captura de imagem")
            break

        rostos = detectarFace(frame, conf_threashold=0.8)

        for (x, y, w, h) in rostos:
            img_rosto = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            try:
                analysis = DeepFace.analyze(img_rosto, actions=["emotion"], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]

                emotion = max(analysis["emotion"], key=analysis["emotion"].get)

                #print da emocao detectada no cmd, para verificacao:
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
                    pasta_pessoa = os.path.join(dir, match)
                    os.makedirs(pasta_pessoa, exist_ok=True)
                    pasta_emocao = os.path.join(pasta_pessoa, emotion)
                    os.makedirs(pasta_emocao, exist_ok=True)
                    nome_arquivo = f"{emotion}_{len(os.listdir(pasta_emocao))+1}.jpg"
                    caminho_arquivo = os.path.join(pasta_emocao, nome_arquivo)

                    cv2.imwrite(caminho_arquivo, img_rosto)
                else:
                    label = "Pessoa nao reconhecida"

                display_text = f"Nome: {label} | Emocao: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            except Exception as e:
                print("Nao foi possivel reconhecer rosto: ", e)

        h_webcam, w_webcam = 400, 460
        webcam_resized = cv2.resize(frame, (w_webcam, h_webcam))

        tela[0:h_webcam, 0:w_webcam] = webcam_resized

        cv2.imshow("Reconhecimento facial", tela)

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
janela.title("Ferramenta facilitadora de criacao de datasets")
janela.geometry("650x400")
janela.configure(bg="#3e4e60")

# Título:
titulo = tk.Label(janela, text="Ferramenta facilitadora\nde criação de datasets", font=("Arial", 24))
titulo.pack(pady=10)
titulo.configure(bg="#3e4e60", fg="white")

# nome usuario:
nomeusu = tk.Label(janela, text="Digite seu nome:", font=("Arial", 16))
nomeusu.pack(pady=10)
nomeusu.configure(bg="#3e4e60", fg="white")

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
botaoTreinamento.pack(pady=10)
botaoTreinamento.configure(bg="#667f98", fg="white")

# Botão Mostrar Emoções:
botaoEmocao = tk.Button(janela, font=("Arial", 16), text="Iniciar Reconhecimento", command=lambda: barraDeCarregamento(reconhecimentoEmocao))
botaoEmocao.pack(pady=10)
botaoEmocao.configure(bg="#667f98", fg="white")

janela.mainloop()
