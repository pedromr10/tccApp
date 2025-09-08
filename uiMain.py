# IMPORTS:
import customtkinter as ctk
import threading
import os, cv2, mss, time
import numpy as np
from deepface import DeepFace

# Configuração do customtkinter (apenas UI)
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Criar dataset, antes de tudo:
dir = "Dataset"
os.makedirs(dir, exist_ok=True)

# Carrega o modelo DNN:
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# FUNCOES:
def mostrarInfos():
    janelaInfo = ctk.CTkToplevel()
    janelaInfo.title("Informações do projeto")
    janelaInfo.geometry("600x350")
    janelaInfo.configure(fg_color="#3e4e60")
    textoInfo = ctk.CTkLabel(janelaInfo, text="Objetivo:\nO projeto tem como objetivo identificar sentimentos e\n expressões faciais de jogadores durante partidas de jogos digitais, por meio da \ncaptura de imagens via webcam. A proposta é disponibilizar uma ferramenta \nque facilite a criação de datasets com dados dos jogadores e das partidas, sem a \nnecessidade de desenvolver jogos próprios para a análise.", font=("Roboto", 16),
                      text_color="white")
    textoInfo.pack(pady=10)
    textoIntegr = ctk.CTkLabel(janelaInfo, text="Integrantes:\n+ Alan Daiki Suga\n+ Gustavo Gomes Barbosa\n+ Pedro Munhoz Rosin", font=("Roboto", 16),
                      text_color="white")
    textoIntegr.pack(pady=10)
    textoOrie = ctk.CTkLabel(janelaInfo, text="Orientador:\n+ Prof. Dr. Fagner de Assis Moura Pimentel", font=("Roboto", 16),
                      text_color="white")
    textoOrie.pack(pady=10)

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

    janelaQtdCapturas = ctk.CTkToplevel()
    janelaQtdCapturas.title("Quantidade de capturas desejadas")
    janelaQtdCapturas.geometry("600x400")
    janelaQtdCapturas.configure(fg_color="#3e4e60")

    ctk.CTkLabel(janelaQtdCapturas, text="Quantidade de capturas: ", font=("Roboto", 36),
                 text_color="white").pack(pady=10)

    qtdInserida = ctk.CTkEntry(janelaQtdCapturas, font=("Roboto", 36), fg_color="#3e4e60", text_color="white")
    qtdInserida.pack(pady=10)

    ctk.CTkButton(janelaQtdCapturas, text="Iniciar", font=("Roboto", 36),
                  command=lambda: barraDeCarregamento(iniciar_captura_de_verdade),
                  fg_color="#3e4e60", text_color="white").pack(pady=10)

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

                    num_arquivo = len([f for f in os.listdir(pasta_emocao) if emotion in f and "tela" not in f]) + 1

                    # Nomes dos arquivos
                    nome_webcam = f"{emotion}_{num_arquivo}.jpg"       # webcam
                    nome_tela = f"tela_{emotion}_{num_arquivo}.jpg"   # tela

                    caminho_webcam = os.path.join(pasta_emocao, nome_webcam)
                    caminho_tela = os.path.join(pasta_emocao, nome_tela)

                    # Salvar webcam e tela sincronizadas
                    cv2.imwrite(caminho_webcam, frame)
                    cv2.imwrite(caminho_tela, tela)

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
    janelaBarraDeCarregamento = ctk.CTkToplevel()
    janelaBarraDeCarregamento.title("Carregando...")
    janelaBarraDeCarregamento.geometry("600x100")
    janelaBarraDeCarregamento.configure(fg_color="#3e4e60")

    ctk.CTkLabel(janelaBarraDeCarregamento, text="Por favor, aguarde alguns instantes...",
                 font=("Roboto", 20), text_color="white").pack(pady=10)

    barra = ctk.CTkProgressBar(janelaBarraDeCarregamento, mode="indeterminate", width=300)
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
janela = ctk.CTk()
janela.title("Ferramenta facilitadora de criacao de datasets")
janela.geometry("650x400")
janela.configure(fg_color="#3e4e60")

# Título:
titulo = ctk.CTkLabel(janela, text="Ferramenta facilitadora\nde criação de datasets", font=("Roboto", 24),
                      text_color="white")
titulo.pack(pady=10)

# nome usuario:
nomeusu = ctk.CTkLabel(janela, text="Digite seu nome:", font=("Roboto", 16), text_color="white")
nomeusu.pack(pady=10)

# Nome do Usuário:
nome = ctk.CTkEntry(janela, font=("Roboto", 20), fg_color="#667f98", text_color="white",width=250)
nome.pack(pady=10)

#Botao Criar Dataset:
botaoDataset = ctk.CTkButton(janela, font=("Roboto", 16), text="Criar Dataset",
                             command=lambda: criarDataset(nome.get()),
                             fg_color="#667f98", text_color="white")
botaoDataset.pack(pady=10)

#botao para treinar o dataset para reconhecimento de perfil (pro deepface saber que vc é vc!!!):
botaoTreinamento = ctk.CTkButton(janela, font=("Roboto", 16), text= "Treinar dataset",
                                 command=lambda: barraDeCarregamento(treinarDataset),
                                 fg_color="#667f98", text_color="white")
botaoTreinamento.pack(pady=10)

# Botão Mostrar Emoções:
botaoEmocao = ctk.CTkButton(janela, font=("Roboto", 16), text="Iniciar Reconhecimento",
                             command=lambda: barraDeCarregamento(reconhecimentoEmocao),
                             fg_color="#667f98", text_color="white")
botaoEmocao.pack(pady=10)

#botao de informacoes:

botaoInfo = ctk.CTkButton(janela, font=("Roboto", 16), text="Informações", fg_color="#667f98", text_color="white", command=lambda: mostrarInfos())
botaoInfo.pack(pady=10)

janela.mainloop()
