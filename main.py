import cv2, os
from deepface import DeepFace

# Diretório base
base_dir = "dataset"
os.makedirs(base_dir, exist_ok=True)

# Exigir nome do usuário
nome_user = input("Digite seu nome: ").strip().lower().replace(" ", "_")

usuario_dir = os.path.join(base_dir, nome_user)
nome_cont = 1
while os.path.exists(usuario_dir):
    usuario_dir = os.path.join(base_dir, f"{nome_user}_{nome_cont}")
    nome_cont += 1

os.makedirs(usuario_dir)

# Inicia a webcam (0 = câmera padrão)
cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair.")

# Contador de cada emoção
cont_emocoes = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tenta detectar emoção no frame
    try:
        # analyze retorna uma lista de resultados (um por rosto)
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Caso múltiplos rostos, pega o primeiro
        if isinstance(results, list):
            emotion = results[0]["dominant_emotion"]
        else:
            emotion = results["dominant_emotion"]

        # Mostra a emoção no frame
        cv2.putText(frame, f"Emocao: {emotion}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # COLOCAR A EMOCAO EM UM ARQUIVO TXT:
        print(emotion)
        with open("emotionResults.txt", "a", encoding="utf-8") as f:
            f.write(emotion + "\n")

        # Criar a pasta da emoção se ela não existir
        emotion_dir = os.path.join(usuario_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        cont_emocoes.setdefault(emotion, 0)

        if(cont_emocoes[emotion] < 100):
            cont_emocoes[emotion] += 1
            # Salvar o frame na pasta de emoção correta
            filename = os.path.join(emotion_dir, f"{emotion}_{cont_emocoes[emotion]}.jpg")
            cv2.imwrite(filename, frame)

    except Exception as e:
        print("Erro na analise:", e)

    # Exibe o vídeo
    cv2.imshow("Webcam - Deteccao de Emocoes", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()