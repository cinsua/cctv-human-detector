'''
python -m venv openvino_env
openvino_env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

'''


import numpy as np
import cv2
from itertools import product
from src.mot_det import MotionDetector
from src.fps import Fps
from src import utils
from src.ia_det import IaDetector
import collections
### PARAMETERS

N_FRAMES_MOV_DET = 10
TRESHOLD_MOV_DET = 40

N_FRAMES_UPDATE_FPS = 30

POSITIVE_DETECTIONS_REQUIRED = 3

CAMERA_DIR = 0 # use RTSP link, or 0 for webcam

###

# Video Capture
video_stream = cv2.VideoCapture(CAMERA_DIR)
ret, first_frame = video_stream.read()

# Init motion detector
mot_det = MotionDetector(first_frame,N_FRAMES_MOV_DET, TRESHOLD_MOV_DET)

# Init fps counter
fps_counter = Fps(N_FRAMES_UPDATE_FPS)

#Init Ia Detector
ia_det = IaDetector()

#INIT DEQUE POSITIVE DETECTIONS REQUIRED to trigger alarms:
stack_events = collections.deque(maxlen=POSITIVE_DETECTIONS_REQUIRED)

while True:
    ret, frame = video_stream.read()
    fps_counter.start_fps()
    #motion detection
    movement_detected = mot_det.process_frame(frame)
    if movement_detected:
        ia_detected = ia_det.process_frame(frame)
        # process intersections

    # reset of bboxes
    if mot_det.bboxes == []:
        ia_det.bboxes = []
    fps_counter.end_fps()
    utils.put_mot_det_rect_in_frame(frame,mot_det.bboxes)
    utils.put_ia_det_rect_in_frame(frame,ia_det.bboxes, ia_det.scores)
    

    # save frame with bbox
    # TO DO 
    if stack_events.count(True)==POSITIVE_DETECTIONS_REQUIRED:
        utils.console_log('ALARM!!')
        utils.beep()
        utils.save_frame_to_img(frame)
        stack_events.clear()
    
    utils.put_fps_in_frame(frame,fps_counter.get_fps_label())

    cv2.imshow('CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()

cv2.destroyAllWindows()
cv2.waitKey(1)



