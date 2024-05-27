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
from src.mot_det_v2 import MotionDetectorV2
from src.fps import Fps
from src import utils
from src.ia_det import IaDetector
from src.bbox import Bbox
import collections
from src.path import PathHandler
from src.detection import Detection
from src.video_capture import VideoCapture
### PARAMETERS

#N_FRAMES_MOV_DET = 10
N_FRAMES_MOV_DET = 5
TRESHOLD_MOV_DET = 40

N_FRAMES_UPDATE_FPS = 30

POSITIVE_DETECTIONS_REQUIRED = 3

#CAMERA_DIR = 0 # use RTSP link, or 0 for webcam
CAMERA_DIR = 'videos/ex7.mp4'

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
def get_frame():
    ret, frame = video_stream.read()
    frame = cv2.resize(frame, dsize=[960,540])
    return ret,frame




ret, first_frame = get_frame()

width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(video_stream.get(cv2.CAP_PROP_FPS))
print(width,height,fps)

frametime = int(500 / fps)

#
#video_capture = VideoCapture(960,540,fps)
video_capture = VideoCapture(640,360,fps)
#video_capture = VideoCapture(352,288,fps)
# Init motion detector
#mot_det = MotionDetector(first_frame,N_FRAMES_MOV_DET, TRESHOLD_MOV_DET)
mot_det = MotionDetectorV2(first_frame,N_FRAMES_MOV_DET, TRESHOLD_MOV_DET)

# Init fps counter
fps_counter = Fps(N_FRAMES_UPDATE_FPS)

#Init Ia Detector
ia_det = IaDetector()

#INIT DEQUE POSITIVE DETECTIONS REQUIRED to trigger alarms:
path_handler = PathHandler(POSITIVE_DETECTIONS_REQUIRED)

frame_buffer = collections.deque(maxlen=int(fps))
#alarm_boxes = []
shadows_confirmed_boxes = []
alarm_boxes_confirmed = []

while True:
    ret, frame = get_frame()
    #frame = cv2.resize(frame, dsize=[960,540])
    fps_counter.start_fps()
    #motion detection
    movement_detected = mot_det.process_frame(frame)
    
    ia_detected = False
    
    # Force to improve mot det
    movement_detected = False
    if movement_detected:
        ia_detected = ia_det.process_frame(frame)
        shadows_confirmed_boxes = [] # should be here?
    
    # process intersections
        frame_detections = []
        if ia_detected:
            
            for ai_box in ia_det.bboxes:
                for mov_box in mot_det.bboxes:
                    coverage, confirmed_box = ai_box.coverage(mov_box)
                    if coverage<COVERAGE_TRESHOLD:
                        continue
                    iou = ai_box.iou(mov_box)
                    if iou < IOU_MAX:
                        continue
                    shadows_confirmed_boxes.append(confirmed_box)                  
                    frame_detections.append(Detection(ai_box,mov_box,confirmed_box))

        path_handler.process_paths(frame_detections)

    # reset of bboxes
    if mot_det.bboxes == []:
        ia_det.bboxes = []
        shadows_confirmed_boxes = []
        alarm_boxes = []
        frame_alarms = []
        #stack_events.clear()
        alarm_boxes_confirmed = []
    fps_counter.end_fps()

    frame=utils.put_mot_det_shadow_in_frame(frame,mot_det.bboxes)
    frame=utils.put_ia_det_shadow_in_frame(frame,ia_det.bboxes)
    frame=utils.put_alarm_det_shadow_in_frame(frame,shadows_confirmed_boxes)

    utils.put_mot_det_rect_in_frame(frame,mot_det.bboxes)
    utils.put_ia_det_rect_in_frame(frame,ia_det.bboxes, ia_det.scores)

    alarm_boxes, trackpoints = path_handler.get_positives()
    
    
    utils.put_alarm_det_rect_in_frame(frame,alarm_boxes)
    utils.put_trackpoinst_in_frame(frame, trackpoints)
    
    # Temporary save frames for testing.
    #if TEMPORARY_TRIGGER:
    #    TEMPORARY_TRIGGER = False
    #    utils.console_log('ALARM!!')
    #    utils.beep()
    #    utils.save_frame_to_img(frame)
        # TO AVOID SUCESIVE TRIGGERS IN LARGE MOTIONS. for TEST ONLY
        # DO NOT CLEAR THE STACK HERE. REMOVE!!! IT CAUSES LOST TRACK ON TARGETS
        #stack_events.clear()

    if len(alarm_boxes)>0:

        video_capture.start_recording()
    
    frame_buffer.append(frame)

    video_capture.process_frame(frame_buffer)
    
    # refactor this
    utils.put_fps_in_frame(frame,fps_counter.get_fps_label())

    cv2.imshow('CCTV', frame)
    if cv2.waitKey(frametime) & 0xFF == ord('q'):
        break

video_stream.release()

cv2.destroyAllWindows()
cv2.waitKey(1)



