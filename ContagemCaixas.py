import cv2
from ultralytics import YOLO
from tkinter import filedialog, Tk

camera = False
escala = 1.5

Tk().withdraw()  # Oculta a janela principal do tkinter

model = YOLO("C:/Users/Ampla Intelligence/Desktop/pastaTreino/my_model1/train/weights/best.pt")

BOX_CLASS_ID = 0
CONF_THRESHOLD = 0.7

cap = None
frame = None

if camera:
    rtsp_url = "rtsp://admin:Ampl@1234@192.168.15.29/cam/realmonitor?channel=1&subtype=1"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    assert cap.isOpened(), "Erro ao abrir a câmera RTSP!"
else:
    img_path = filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
    if img_path:
        frame = cv2.imread(img_path)
        CONF_THRESHOLD -= 0.2
    else:
        print("Nenhuma imagem selecionada. Encerrando...")
        exit()

def redimensionar(imagem, fator):
    altura = int(imagem.shape[0] * fator)
    largura = int(imagem.shape[1] * fator)
    return cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_LINEAR)

def detectar_e_mostrar(imagem):
    frame_redim = redimensionar(imagem, escala)
    results = model(frame_redim, conf=CONF_THRESHOLD)
    boxes = results[0].boxes
    box_count = 0

    for i in range(len(boxes)):
        class_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)

        if class_id == BOX_CLASS_ID and conf >= CONF_THRESHOLD:
            box_count += 1
            cv2.rectangle(frame_redim, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame_redim, f"Caixa: {conf*100:.1f}%", (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame_redim, f"Caixas detectadas: {box_count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # Mostra a imagem processada em outra janela (sem bloquear o vídeo ao vivo)
    cv2.imshow("Última Foto Processada", frame_redim)

if cap is not None:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível ler frame da câmera.")
            break

        frame_redim = redimensionar(frame, escala)
        cv2.imshow("Vídeo Ao Vivo - Pressione 's' para tirar foto | 'ESC' para sair", frame_redim)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Tecla ESC
            break
        elif key == ord('s'):  # Tecla 's' para tirar foto
            detectar_e_mostrar(frame)

# Se foi imagem local
if cap is None and frame is not None:
    detectar_e_mostrar(frame)
    cv2.waitKey(0)

if cap:
    cap.release()
cv2.destroyAllWindows()
