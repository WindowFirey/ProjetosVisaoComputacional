from collections import deque
import pyrealsense2 as rs
import numpy as np
import statistics
import cv2
 
def DesenhaInformacoes(img, best_rect, cx,cy,xCamera,yCamera,dYCentro,dXCentro,dCentro,area,avg_depth,width,height):
    cv2.drawContours(img, [best_rect], -1, (0, 255, 0), 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

    #print(media['dYCentro'])
    cv2.line(img, (xCamera, yCamera), (xCamera, cy), (255, 0, 0), 2)
    cv2.line(img, (cx, cy), (xCamera, cy), (255, 0, 0), 2)
    cv2.line(img, (xCamera, yCamera), (cx, cy), (0, 255, 255), 2)

    cv2.putText(img, f"Distancia do X: {dYCentro:.2f} cm", (0,yCamera * 2 - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f"Distancia do Y: {dXCentro:.2f} cm", (0,yCamera * 2 - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img, f"Distancia do Centro: {dCentro:.2f} cm", (0,yCamera * 2 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(img, f"Area: {area:.2f} cm2", (0,yCamera * 2 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, f"Altura: {avg_depth:.2f} cm", (0,yCamera * 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, f"Largura: {width:.2f} cm", (0,yCamera * 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(img, f"Comprimento: {height:.2f} cm", (0,yCamera * 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


escala = 100 #Centimetro
xCamera = 640
yCamera = 480

# Inicializa a câmera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, xCamera, yCamera, rs.format.z16, 30)
config.enable_stream(rs.stream.color, xCamera, yCamera, rs.format.bgr8, 30)
pipeline.start(config)

xCamera = xCamera //2
yCamera = yCamera //2
align = rs.align(rs.stream.color)

registros = deque(maxlen=100)
media = {}


i = 0

try:  
    while True:
        # Captura os frames
        
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Conversão para arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Pré-processamento
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 150)

        kernel = np.ones((3,3),np.uint8)
        imgDilate = cv2.dilate(edges,kernel,iterations=3)
        imgErode = cv2.erode(imgDilate,kernel,iterations=2)

        cv2.imshow("contours", imgErode)

        # Contornos
        contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        best_rect = None

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area > 500 and area > largest_area:  # ← ajuste aqui o mínimo
                    largest_area = area
                    best_rect = approx

            if best_rect is not None:
                pts = best_rect.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]
                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]




    
                (tl, tr, br, bl) = rect
                cx = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
                cy = int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
                width = np.linalg.norm(tr - tl)
                height = np.linalg.norm(bl - tl)

    
                # Profundidade média
                depth_values = []
                for point in rect:
                    x, y = int(point[0]), int(point[1])
                    d = depth_frame.get_distance(x, y)
                    if d > 0:
                        depth_values.append(d)
                if not depth_values:
                    continue
                avg_depth = np.mean(depth_values)
    
                # Parâmetros da câmera
                fx = 615.0 / escala
                fy = 615.0 / escala
    
                width = (width * avg_depth) / fx
                height = (height * avg_depth) / fy

                area = width * height
                avg_depth = avg_depth * escala

                dCentro = (((xCamera - cx)**2 + (yCamera - cy)**2)**0.5)/escala * 10

                dXCentro = (((xCamera - xCamera)**2 + (yCamera - cy)**2)**0.5)/escala * 10
                dYCentro = (((cx - xCamera)**2 + (cy - cy)**2)**0.5)/escala * 10
                
                i+=1
                dicionarioLimpo = False

                novo_registro = {
                    "rect0": rect[0][0],
                    "rect1": rect[0][1],
                    'avg_depth': avg_depth,
                    'width': width,
                    'height': height,
                    'area': area,
                    'dCentro': dCentro,
                    'dXCentro': dXCentro,
                    'dYCentro': dYCentro
                }
                if registros:
                    #verifica diferenca entre valores, se algum for muito alto reseta os valores
                    rect0_anterior = registros[-1]['rect0']
                    rect1_anterior = registros[-1]['rect1']
                    avg_depth_anterior = registros[-1]['avg_depth']

                    if (abs(rect[0] - rect0_anterior) > 50).all() or (abs(rect[1] - rect1_anterior) > 50).all() or abs(avg_depth - avg_depth_anterior) > 50:
                        registros.clear()
                        media.clear()
                        registros.append(novo_registro)
                        dicionarioLimpo = True
                        i = 0

                if not dicionarioLimpo:
                    registros.append(novo_registro)
                    if len(registros) == 100:
                        media = {}
                        for chave in registros[0].keys():
                            media[chave] = round(statistics.mean(r[chave] for r in registros), 2)   


                        cv2.imwrite(f"FOTOS/img1.png", color_image)
                        img =  cv2.imread('FOTOS/img1.png')


                        # Desenha informações
                        DesenhaInformacoes(img, best_rect, 
                                            cx, cy, 
                                            xCamera, yCamera, 
                                            media['dYCentro'], media['dXCentro'], media['dCentro'], 
                                            media['area'], media['avg_depth'], media['width'], media['height'])
                            

                        cv2.imshow("foto",img)
                        registros.clear()
                        media.clear()

                        
                        print("="*80)

                print(i)
                

        cv2.circle(color_image, (xCamera, yCamera), 5, (0, 255, 255), -1)  
        cv2.imshow("Video",color_image)

        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()