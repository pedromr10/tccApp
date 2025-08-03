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


#TREINAMENTO DE DATASET:
def treinar_dataset():
    agrupamento={}
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
                    print("Erro ao treinar imagens")
    return agrupamento

#Reconhecimento facial, usando DeepFace:

def reconhecimento_facial(agrupamentos):
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
                # CORREÇÃO AQUI: era "analyse", o correto é "analyze"
                analysis = DeepFace.analyze(img_rosto, actions=["age", "emotion"], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]

                age = analysis.get("age", "N/A")
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

                display_text = f"{label}, Idade: {int(age)}, Emocao: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            except Exception as e:
                print("Nao foi possivel reconhecer rosto:", e)

        cv2.imshow("Reconhecimento facial", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()


#OUTPUT:
if __name__ == "__main__":
    print("1. Criar dataset\n2. Treinar dataset de rosto\n3. Reconhecimento facial")
    
    escolha = int(input("Escolha a opcao: "))
    if escolha == 1:
        name = input("Digite seu nome:")
        criarDataset(name)
    elif escolha == 2:
        embedding = treinar_dataset()
        np.save("embedding.npy", embedding )
    if escolha == 3:
        if os.path.exists("embedding.npy"):
            embedding = np.load("embedding.npy", allow_pickle = True).item()
            reconhecimento_facial(embedding)
        else:
            print("arquivo nao")


#OBS: cascade tirado de: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml