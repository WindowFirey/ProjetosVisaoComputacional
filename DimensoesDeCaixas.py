import pyrealsense2 as rs
import numpy as np
import cv2
import time

########################### Configurações ###########################
escala = 100  # Centímetro
alturaCamera = 1.05 * escala # em centímetros
xCamera = 640
yCamera = 480
densidade = 0.6
padding_warp = 20  # Padding para o warp
#####################################################################

# Funções auxiliares
def reorder(myPoints):
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def warpImg(img, points, w, h, pad=padding_warp):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]
    return imgWarp

# Inicializa a câmera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, xCamera, yCamera, rs.format.z16, 30)
config.enable_stream(rs.stream.color, xCamera, yCamera, rs.format.bgr8, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

def capturar_foto():
    """Captura uma foto e retorna os frames alinhados"""
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    return depth_frame, color_frame

def processar_imagem(color_frame, depth_frame):
    """Processa a imagem para tentar detectar a caixa"""
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    # Pré-processamento
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 150)

    kernel = np.ones((3,3), np.uint8)
    imgDilate = cv2.dilate(edges, kernel, iterations=3)
    imgErode = cv2.erode(imgDilate, kernel, iterations=2)

    # Contornos
    contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    best_rect = None

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 10000 and area > largest_area:
                largest_area = area
                best_rect = approx
    
    return best_rect, color_image, depth_image, depth_frame, edges, imgErode

def calcular_dimensoes(best_rect, depth_frame, color_image, depth_image):
    """Calcula as dimensões da caixa e desenha na imagem"""
    pts = best_rect.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    width = np.linalg.norm(tr - tl)
    height = np.linalg.norm(bl - tl)

    # Calcula dimensões do warp
    w = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    h = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))

    # Aplica warp na imagem colorida e de profundidade
    img_warped = warpImg(color_image, best_rect, w, h)
    depth_warped = warpImg(depth_image, best_rect, w, h)
    
    # Profundidade média na área warped
    depth_values = depth_warped[depth_warped > 0]
    if len(depth_values) == 0:
        return None, None, None, None, None
    
    avg_depth = np.mean(depth_values)

    # Parâmetros da câmera
    fx = 615.0 / escala
    fy = 615.0 / escala

    width_m = ((width * avg_depth) / fx) / (escala * 10)
    height_m = ((height * avg_depth) / fy) / (escala * 10)
    avg_depth_cm = avg_depth / escala * 10

    return color_image, img_warped, width_m, height_m, avg_depth_cm

try:
    while True:
        sucesso = False
        
        while True:
            # Captura uma foto
            depth_frame, color_frame = capturar_foto()
            if not depth_frame or not color_frame:
                print("Falha ao capturar frame. Tentando novamente...")
                time.sleep(0.1)
                continue
            
            # Processa a imagem
            best_rect, color_image, depth_image, depth_frame, edges, img_processed = processar_imagem(color_frame, depth_frame)
            
            # Mostra visualização ao vivo
            cv2.imshow("Visao da Camera", color_image)
            
            if best_rect is not None and not sucesso:
                print("Caixa detectada com sucesso!")
                img_resultado, img_warped, comprimento, largura, profundidade = calcular_dimensoes(best_rect, depth_frame, color_image, depth_image)
                if img_resultado is not None:
                    sucesso = True
                    # Desenha as informações na imagem
                    cv2.drawContours(color_image, [best_rect], -1, (0, 255, 0), 2)

                    massa = ((comprimento * largura * (alturaCamera - profundidade))/1000000) * 380

                    cv2.putText(color_image, f"Peso: {massa:.2f} quilos", (20, yCamera - 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(color_image, f"Altura: {(alturaCamera - profundidade):.2f} cm", (20, yCamera - 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(color_image, f"Largura: {comprimento:.2f} cm", (20, yCamera - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(color_image, f"Comprimento: {largura:.2f} cm", (20, yCamera - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    
                    # Mostra os resultados
                    cv2.imshow("Resultado - Dimensoes da Caixa", color_image)
                    
                    print(f"Largura media da caixa: {comprimento:.2f} cm")
                    print(f"Comprimento media da caixa: {largura:.2f} cm")
                    print(f"Altura media da caixa: {(alturaCamera - profundidade):.2f} cm")
                    print("\nPressione R para detectar novamente\n")
            # Tecla 'q' para sair ou 'r' para recomeçar
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                pipeline.stop()
                cv2.destroyAllWindows()
                exit()
            elif key & 0xFF == ord('r'):
                print("\nReiniciando busca pela caixa...")
                cv2.destroyWindow("Resultado - Dimensoes da Caixa")
                break
            
            # Se já encontrou a caixa, não precisa continuar processando até pressionar 'r'
            if sucesso:
                continue
                
finally:
    pipeline.stop()
    cv2.destroyAllWindows()