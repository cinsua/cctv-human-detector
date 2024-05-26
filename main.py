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
#CAMERA_DIR = 'vid/ex2.mp4'

COVERAGE_TRESHOLD = 0.5
IOU_MAX = 0.4

###
'''
STATE of this project:
- ALL can change
- Right now i am testing the viability in real world scenarios.
- Why? because traditional alarms (especially PIR sensors) have problems with cats, and i have 5, and big windows. So i need movement detection and get rid of cat false alarm.
- The goal here is have one old pc as server, an arduino for sirens/remote controls/(and sms messages when internet is not available), 4 or more cameras conected to LAN. And have the funtionality of camera recording and Home Alarm in one system
'''

'''
TODO
- all boxes should be instances of Bbox. Improve Bbox capabilities
- refactor the nasty nesting code with classes
- include @click params or similar
- add descriptions to all params
- improve detection. maybe with distance center-to-center instead of overlaping/coverage, explore other options
- add line tracking to the detection
- add saving video capabilities
- add AlertOutput class.. all the actions reports should be handled there. like trigger alarm, tampering, background compromise, cammera disconection, etc
- update readme with examples vid, gifs, and better explanation of what i do and why
- option to load params from a cfg file
- add states/modes. ex: 
    ALARM mode: movement + AI to fire alarm, tampering(more than 50% of view movement detected), background profund change (like blocking the sight of the camera).
    WATCH mode: deactivate AI confirmation. start saving clips of the prolonged movements detected
    etc
Low priority:
- get another ai model. the one i am using here is relatively heavy, and i am not even using the segmentation part
- provide a server file example, or ideas for the real use of this

'''



# Video Capture
video_stream = cv2.VideoCapture(CAMERA_DIR)
ret, first_frame = video_stream.read()
#first_frame = cv2.resize(first_frame, dsize=[960,540])

# Init motion detector
mot_det = MotionDetector(first_frame,N_FRAMES_MOV_DET, TRESHOLD_MOV_DET)

# Init fps counter
fps_counter = Fps(N_FRAMES_UPDATE_FPS)

#Init Ia Detector
ia_det = IaDetector()

#INIT DEQUE POSITIVE DETECTIONS REQUIRED to trigger alarms:
stack_events = collections.deque(maxlen=POSITIVE_DETECTIONS_REQUIRED)
#alarm_boxes = []
frame_alarms = []
alarm_boxes_confirmed = []

TEMPORARY_TRIGGER = False

while True:
    ret, frame = video_stream.read()
    #frame = cv2.resize(frame, dsize=[960,540])
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
                    previous_box = bbox
                    for index,prev_frame_alarms in enumerate(stack_events):
                        accepted_box = None
                        accepted_coverage=0
                        if index == 0:
                            continue
                        if index>trigger_counter:
                            break
                        
                        for ab in prev_frame_alarms:
                            x, y, x_max, y_max = ab
                            abbox = Bbox(x,y,x_max,y_max)
                            coverage, _ = previous_box.coverage(abbox)
                            # TODO trace movement from feets
                            if coverage>0 and trigger_counter >= index:
                                if accepted_coverage<coverage:
                                    # this is JS level of nesting madness
                                    # when i get it working im promise that i will refactor this
                                    accepted_coverage = coverage
                                    accepted_box = abbox
                                if trigger_counter == index:
                                    trigger_counter += 1
                        if accepted_coverage>0:
                            previous_box = accepted_box

                    if trigger_counter>=POSITIVE_DETECTIONS_REQUIRED:
                        #print('ALARM', i, trigger_counter)
                        alarm_boxes_confirmed.append([x1, y1, x2, y2,trigger_counter])
                        TEMPORARY_TRIGGER = True



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
    
    # Temporary save frames for testing.
    if TEMPORARY_TRIGGER:
        TEMPORARY_TRIGGER = False
        utils.console_log('ALARM!!')
        utils.beep()
        utils.save_frame_to_img(frame)
        # TO AVOID SUCESIVE TRIGGERS IN LARGE MOTIONS. for TEST ONLY
        # DO NOT CLEAR THE STACK HERE. REMOVE!!! IT CAUSES LOST TRACK ON TARGETS
        stack_events.clear()


        
    
    utils.put_fps_in_frame(frame,fps_counter.get_fps_label())

    cv2.imshow('CCTV', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.release()

cv2.destroyAllWindows()
cv2.waitKey(1)



