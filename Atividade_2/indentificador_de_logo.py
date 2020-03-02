# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:12:30 2020

@author: Fernando
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Indentificar o logo do insper - começo
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

def find_homography_draw_box(kp1, kp2, img_cena):
    
    out = img_cena.copy()
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    # Esta transformação é chamada de homografia 
    # Para saber mais veja 
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()


    
    h,w = img_original.shape
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
    dst = cv2.perspectiveTransform(pts,M)


    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
    return img2b

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    
    # Número mínimo de pontos correspondentes
    MIN_MATCH_COUNT = 7
    
    # Imagem a ser detectada 
    insper_bgr = cv2.imread('logo.png')
    insper_gray = cv2.cvtColor(insper_bgr, cv2.COLOR_BGR2GRAY)
    
    cena_bgr = frame # Imagem do cenario
    original_bgr = insper_bgr
    
    # Versões RGB das imagens, para plot
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    cena_rgb = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2RGB)
    
    # Versões grayscale para feature matching
    img_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    img_cena = cv2.cvtColor(cena_bgr, cv2.COLOR_BGR2GRAY)
    
    framed = None

    # Imagem de saída
    out = cena_rgb.copy()
    
    
    # Cria o detector BRISK
    brisk = cv2.BRISK_create()
    
    # Encontra os pontos únicos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(img_original ,None)
    kp2, des2 = brisk.detectAndCompute(img_cena,None)
    
    # Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2,k=2)
    
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        # Separa os bons matches na origem e no destino
        print("Matches found")
        img3 = cv2.drawMatches(original_rgb,kp1,cena_rgb,kp2, good, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img4 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        framed = find_homography_draw_box(kp1, kp2, frame)
        #final = img4 + framed
        cv2.imshow('color',framed )
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        img3 = cv2.drawMatches(original_rgb,kp1,cena_rgb,kp2, good, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img4 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        cv2.imshow('color', frame)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()