import cv2
from deepface import DeepFace

# Inicia a webcam (0 = câmera padrão)
cap = cv2.VideoCapture(0)

print("Pressione 'q' para sair.")

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