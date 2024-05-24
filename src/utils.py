import cv2
import datetime
from playsound import playsound
import _thread
import numpy as np
# FONT_HERSHEY_COMPLEX
# cv2.putText(img,'Hack Projects',(10,500), font, 1,(255,255,255),2, cv2.LINE_AA)

font=cv2.FONT_HERSHEY_COMPLEX
font_scale=0.4
font_thickness=1
text_color=(255, 255, 255)

def put_mot_det_rect_in_frame(frame, bbox):
    # text in bbox left
    text_color_bg = (0,200,0)
    for b in bbox:
        x, y, w, h = b
        frame = cv2.rectangle(frame, (x, y), (x+w,y+h), text_color_bg,1)
        text_size, _ = cv2.getTextSize('Movement', font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x,y), (x + text_w + 4, y + text_h + 4), text_color_bg, -1)
        cv2.putText(frame, 'Movement', (x+2, y+2+ text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        #draw_text(frame,'MOV_DET',pos=(x,y),text_color=(255,255,255),text_color_bg=(0,0,200))
    return frame

def put_ia_det_rect_in_frame(frame, bboxes, scores):
    text_color_bg = (200,0,0)
    for index, b in enumerate(bboxes):
        x, y, x_max, y_max = b
        frame = cv2.rectangle(frame, (x, y), (x_max,y_max),text_color_bg,1)
        text_size, _ = cv2.getTextSize(scores[index], font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x_max-text_w-4,y), (x_max, y + text_h + 4), text_color_bg, -1)
        cv2.putText(frame, scores[index], (x_max-text_w-2, y+2+ text_h), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return frame

def put_alarm_det_rect_in_frame(frame, bboxes):
    text_color_bg = (0,0,200)
    for index, b in enumerate(bboxes):
        x, y, x_max, y_max = b
        frame = cv2.rectangle(frame, (x, y), (x_max,y_max),text_color_bg,1)
        text_size, _ = cv2.getTextSize('ALARM', font, font_scale+0.1, font_thickness+1)
        text_w, text_h = text_size
        cv2.rectangle(frame, (x,y_max - text_h - 4), (x + text_w + 4, y_max), text_color_bg, -1)
        cv2.putText(frame, 'ALARM', (x+2, y_max-2), font, font_scale+0.1, text_color, font_thickness, cv2.LINE_AA)
    
    return frame

def put_alarm_det_shadow_in_frame(frame, bboxes):
    shapes = np.zeros_like(frame, np.uint8)
    for index, b in enumerate(bboxes):
        x, y, x_max, y_max = b
        cv2.rectangle(shapes, (x, y), (x_max,y_max), (0, 0, 200),cv2.FILLED)
    out = frame.copy()
    alpha = 0.5
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
    return out

def put_mot_det_shadow_in_frame(frame, bbox):
    shapes = np.zeros_like(frame, np.uint8)
    for b in bbox:
        x, y, w, h = b
        cv2.rectangle(shapes, (x, y), (x+w,y+h), (0, 200, 0), cv2.FILLED)
    out = frame.copy()
    alpha = 0.7
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
    return out

def put_ia_det_shadow_in_frame(frame, bboxes):
    shapes = np.zeros_like(frame, np.uint8)
    for index, b in enumerate(bboxes):
        x, y, x_max, y_max = b
        cv2.rectangle(shapes, (x, y), (x_max,y_max), (200, 0, 0),cv2.FILLED)
    out = frame.copy()
    alpha = 0.7
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
    return out

def console_log(text):
    now = datetime.datetime.now()
    print(f'[{now.hour}:{now.minute}:{now.second}] {text}')

def play_audio():
    playsound('./beep.mp3')

def beep():
    _thread.start_new_thread( play_audio, () )
    #playsound('./beep.mp3')

def save_frame_to_img(frame):
    img = cv2.resize(frame, dsize=[640,360])
    now = datetime.datetime.now()
    #dt_string = now.strftime("%m-%d")
    dt_string = now.strftime("%m-%d %H.%M.%S")
    #print(dt_string)
    #'./images/'
    cv2.imwrite('images/'+dt_string+'.jpg', img)



def put_fps_in_frame(frame, text, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    position = (10, 50)  # top left corner position

    # Add the text to the image
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

    return frame
