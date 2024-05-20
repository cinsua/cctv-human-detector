#import the necessary packages
import numpy as np
import cv2
from itertools import product
from src.mot_det import MotionDetector
from src.fps import Fps
from src import utils
from src.ia_det import IaDetector

### PARAMETERS

N_FRAMES_MOV_DET = 10
TRESHOLD_MOV_DET = 25

N_FRAMES_UPDATE_FPS = 30

CAMERA_DIR = 0 # use RTSP link, or 0 for webcam

###
mot_det = MotionDetector(N_FRAMES_MOV_DET, TRESHOLD_MOV_DET)
# Video Capture
video_stream = cv2.VideoCapture(CAMERA_DIR)
ret, first_frame = video_stream.read()

# Set background
# it will update once every N_FRAMES_MOV_DET
mot_det.set_background(first_frame)

# Init fps counter
fps_counter = Fps(N_FRAMES_UPDATE_FPS)

#Init Ia Detector
ia_det = IaDetector()

while True:
    ret, frame = video_stream.read()
    fps_counter.start_fps()
    #motion detection
    result = mot_det.process_frame(frame)

    if result:
        #process bboxes with ia
        res_ia = ia_det.process_bboxes(frame,mot_det.bboxes)
        if res_ia == False:
            ia_det.bboxes = []
        #print(res_ia)
    if mot_det.bboxes == []:
        ia_det.bboxes = []

    fps_counter.end_fps()
    utils.put_mot_det_rect_in_frame(frame,mot_det.bboxes)
    utils.put_ia_det_rect_in_frame(frame,ia_det.bboxes)
    utils.put_fps_in_frame(frame,fps_counter.get_fps_label())
    
    cv2.imshow('Streaming', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()

cv2.destroyAllWindows()
cv2.waitKey(1)



