#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import math

def centro(lista_circulos):
    lista = list()
    c = 0
    while c < 1:
        try:
            c = 1
            lista_circulos = np.array(lista_circulos).tolist()
            for circulo in lista_circulos:
                lista.append(circulo)
                
        except:
            c = 1
            continue
    a = 0
    while a < 1:
        try:  
            cordenadas = []
            a = 1
            x_1 = lista[0][0][0]            
            x_2 = lista[0][1][0]
            
            y_1 = lista[0][0][1]
            y_2 = lista[0][1][1]
            
            cordenadas.append(x_1)
            cordenadas.append(x_2)
            cordenadas.append(y_1)
            cordenadas.append(y_2)
                        
            termo_x = abs(((x_1 - x_2)^2))
            termo_y = abs(((y_1 - y_2)^2))
            
            distancia = math.sqrt(termo_x+termo_y)
            
            return [distancia, cordenadas]
        
        except:
            
            a = 1
            return ["sem circulo"]
            continue


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

        
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect the edges present in the image
    bordas = auto_canny(blur)
    
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    ########################## Meu código
    cor_menor_azul = np.array([80, 45, 50])
    cor_maior_azul = np.array([106, 255, 255])
    mask = cv2.inRange(hsv, cor_menor_azul, cor_maior_azul)
    
    
    cor_menor_vermelho = np.array([145, 50, 50])
    cor_maior_vermelho = np.array([179, 255, 255])
    mask_vermelho = cv2.inRange(hsv, cor_menor_vermelho, cor_maior_vermelho)
    
    #mascara = mask_vermelho
    mascara = mask + mask_vermelho
    
    img_1 = cv2.bitwise_or(frame, frame, mask=mascara)
    im = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    
    circles = []
    circles = None
    circles=cv2.HoughCircles(im,cv2.HOUGH_GRADIENT,2,70,param1=50,param2=40,minRadius=20,maxRadius=40)
    
    if circles is not None:        
        circles = np.uint16(np.around(circles)).astype("int")
        for i in circles[0,:]:
            # draw the outer circle
            # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(bordas_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(bordas_color,(i[0],i[1]),2,(0,0,255),3)
    
    dist_centro = centro(circles)[0]
    H = 14
    f = 572
    if dist_centro == "sem circulo":
        D = "Sem distância"
    
    
    else:
        Ds = f*H/dist_centro
        D = Ds*(37/490)
        x_1 = centro(circles)[1][0]
        x_2 = centro(circles)[1][1]
        y_1 = centro(circles)[1][2]
        y_2 = centro(circles)[1][3]
        
        cv2.line(img_1, (x_1,y_1), (x_2,y_2), (0, 255, 0), thickness=3, lineType=8)
        
        termoy = abs(y_1-y_2)
        termox = abs(x_1-x_2)
        angulo = math.atan2(termoy,termox)
        angulo = math.degrees(angulo)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(bordas_color,"Angulo: {}".format(angulo),(0,150), font, (0.75),(255,255,255),2,cv2.LINE_AA)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_1, "Distancia: {}cm".format(D) ,(50,200), font, (0.5),(255,255,255),2,cv2.LINE_AA)
    cv2.putText(bordas_color,'Press q to quit',(0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    
    misturando = img_1 + bordas_color
    cv2.imshow('color', misturando)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()