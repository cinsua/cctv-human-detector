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
from src.intersection import Bbox
import collections
### PARAMETERS

N_FRAMES_MOV_DET = 6
TRESHOLD_MOV_DET = 40

N_FRAMES_UPDATE_FPS = 30

POSITIVE_DETECTIONS_REQUIRED = 3

CAMERA_DIR = 0 # use RTSP link, or 0 for webcam

COVERAGE_TRESHOLD = 0.5
IOU_MAX = 0.4

###

# Video Capture
video_stream = cv2.VideoCapture(CAMERA_DIR)
ret, first_frame = video_stream.read()
first_frame = cv2.resize(first_frame, dsize=[960,540])

# Init motion detector
mot_det = MotionDetector(first_frame,N_FRAMES_MOV_DET, TRESHOLD_MOV_DET)

# Init fps counter
fps_counter = Fps(N_FRAMES_UPDATE_FPS)

#Init Ia Detector
ia_det = IaDetector()

#INIT DEQUE POSITIVE DETECTIONS REQUIRED to trigger alarms:
stack_events = collections.deque(maxlen=5)
#alarm_boxes = []
frame_alarms = []
alarm_boxes_confirmed = []

while True:
    ret, frame = video_stream.read()
    frame = cv2.resize(frame, dsize=[960,540])
    fps_counter.start_fps()
    #motion detection
    movement_detected = mot_det.process_frame(frame)
    if movement_detected:
        ia_detected = ia_det.process_frame(frame)
        # process intersections
        if ia_detected:
            mot_det_bboxes = []
            ia_det_bboxes = []
            for b in mot_det.bboxes:
                x, y, w, h = b
                bbox = Bbox(x,y,x+w,y+h)
                mot_det_bboxes.append(bbox)
            for b in ia_det.bboxes:
                x, y, x_max, y_max = b
                bbox = Bbox(x,y,x_max,y_max)
                ia_det_bboxes.append(bbox)
            #alarm_boxes = []
            frame_alarms = []
            for box in ia_det_bboxes:
                for mov_box in mot_det_bboxes:
                    
                    coverage, alarm_box = box.coverage(mov_box)
                    #print(coverage)
                    if coverage>COVERAGE_TRESHOLD:
                        iou = box.iou(mov_box)
                        if iou > IOU_MAX:
                            # TODO write better code, this nesting is insane
                            frame_alarms.append(alarm_box)
                            #print('ALARM', alarm_box)
            if len(frame_alarms)>0:
                stack_events.appendleft(frame_alarms)

            # this is horrible code, but works.
            if len(stack_events)>=POSITIVE_DETECTIONS_REQUIRED:
                alarm_boxes_confirmed = []
                #index = len(stack_events) -1
                for i,alarm_box in enumerate(stack_events[0]):
                    #print('----------starting alarm',i)
                    x1, y1, x2, y2 = alarm_box
                    bbox = Bbox(x,y,x_max,y_max)
                    trigger_counter = 1
                    for index,prev_frame_alarms in enumerate(stack_events):
                        if index == 0:
                            continue
                        for ab in prev_frame_alarms:
                            x, y, x_max, y_max = ab
                            abbox = Bbox(x,y,x_max,y_max)
                            coverage, _ = bbox.coverage(abbox)
                            if coverage>0 and trigger_counter == index:
                                trigger_counter += 1
                    if trigger_counter-1>=POSITIVE_DETECTIONS_REQUIRED:
                        #print('ALARM', i, trigger_counter)
                        alarm_boxes_confirmed.append([x1, y1, x2, y2,trigger_counter])



    # reset of bboxes
    if mot_det.bboxes == []:
        ia_det.bboxes = []
        alarm_boxes = []
        frame_alarms = []
        stack_events.clear()
        alarm_boxes_confirmed = []
    fps_counter.end_fps()

    frame=utils.put_mot_det_shadow_in_frame(frame,mot_det.bboxes)
    frame=utils.put_ia_det_shadow_in_frame(frame,ia_det.bboxes)
    frame=utils.put_alarm_det_shadow_in_frame(frame,frame_alarms)

    utils.put_mot_det_rect_in_frame(frame,mot_det.bboxes)
    utils.put_ia_det_rect_in_frame(frame,ia_det.bboxes, ia_det.scores)
    utils.put_alarm_det_rect_in_frame(frame,alarm_boxes_confirmed)
    

    # save frame with bbox
    # TO DO 
    #if stack_events.count(True)==POSITIVE_DETECTIONS_REQUIRED:
    #    utils.console_log('ALARM!!')
    #    utils.beep()
    #    utils.save_frame_to_img(frame)
    #    #stack_events.clear()
    



        
    
    utils.put_fps_in_frame(frame,fps_counter.get_fps_label())

    cv2.imshow('CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()

cv2.destroyAllWindows()
cv2.waitKey(1)



