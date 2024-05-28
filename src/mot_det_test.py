import cv2
from src.bbox import Bbox
import numpy as np
import time

'''
1) 
frames t1 t2 t3
diferencias t1-t2 t2-t3 ver entre ambos resultados. aplicar malla movdet2 . deberia tener poco ruido
opcional: low res

2) 
createBackgroundSubtractorMOG/2 sobre low res
compararse entre si en un stack de 3

3)
combinar 1) y 2)

4) volver a probar otros modelos de ia? usando recortes? modelo actual presenta problemas a distancia > 6-8m

5) ver de entrenar una ia - probar yolov10n

6) los path podrian utilizar distancia centro a centro similares? 
quizas usando solo centro a centro de movimiento y confirmando colision con ia, tipo max_distance_ctoc = height_movement en linea recta horizontal

'''

class MotionDetectorTest:
    def __init__(self,frame, n_frames = 10, treshold = 25) -> None:
        #print('INIT---------------------------------------------')
        #t0 = time.perf_counter()
        self.backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        #t1 = time.perf_counter()
        self.background_subtr_method = cv2.bgsegm.createBackgroundSubtractorMOG()
        #t2 = time.perf_counter()
        self.background_subtr_method2 = cv2.bgsegm.createBackgroundSubtractorGSOC()
        #t3 = time.perf_counter()
        #print(t1-t0)
        #print(t2-t1)
        #print(t3-t2)
    
    def _reset(self):
        self.frame_counter = 0
    
    def _preprocess(self,frame):
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #f = cv2.GaussianBlur(f, (21,21), 0)
        return f
    
    def process_frame(self,frame):
        #print('FRAME---------------------------------------------')
        #t0 = time.perf_counter()
        fgMask = self.backSub.apply(frame)
        #t1 = time.perf_counter()
        foreground_mask1 = self.background_subtr_method.apply(frame)
        #t2 = time.perf_counter()
        foreground_mask = self.background_subtr_method2.apply(frame)
        #t3 = time.perf_counter()
        cv2.imshow('delta', foreground_mask1)
        #print(t1-t0)
        #print(t2-t1)
        #print(t3-t2)
