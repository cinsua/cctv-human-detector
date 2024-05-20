import cv2


def put_fps_in_frame(frame, text, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    position = (10, 50)  # top left corner position

    # Add the text to the image
    cv2.putText(frame, text, position, font, font_scale, color, thickness)

    return frame

def put_mot_det_rect_in_frame(frame, bbox):
    for b in bbox:
        x, y, x_max, y_max = b
        frame = cv2.rectangle(frame, (x, y), (x_max,y_max), (0, 0, 200),1)
    
    return frame
