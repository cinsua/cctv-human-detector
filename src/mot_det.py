import cv2

class MotionDetector:
    def __init__(self, n_frames = 10, treshold = 25) -> None:
        self.bboxes = []
        self.n_frames = n_frames
        self.treshold =treshold
        self._reset()
    
    def _reset(self):
        self.frame_counter = 0
    
    def _preprocess(self,frame):
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f = cv2.GaussianBlur(f, (21,21), 0)
        return f

    def set_background(self, frame):
        self.previus_frame = self._preprocess(frame)
    
    def process_frame(self,frame):
        self.frame_counter+=1
        if (self.frame_counter%self.n_frames == 0):
            gray = self._preprocess(frame)
            delta = cv2.absdiff(gray, self.previus_frame)
            delta_thresh = cv2.threshold(delta, self.treshold, 255, cv2.THRESH_BINARY)[1]
            delta_thresh_dilated = cv2.dilate(delta_thresh, None, iterations=5)
            contours, hierarchy = cv2.findContours(delta_thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_contour_area = 500  # Define your minimum area threshold
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            # Get boxes 
            bboxes = []
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # little trick to avoid rectangles with no overlapping dissapear
                bboxes.append([x,y,w,h])
                bboxes.append([x,y,w,h])

            # group boxes overlapped
            bboxes, weights = cv2.groupRectangles(bboxes, groupThreshold=1, eps=2)

            # Transform boxes in bigger squares
            self.bboxes=[]
            for b in bboxes:
                x, y, w, h = b
                d = max([w,h])
                x1 = x - int(d/4)
                y1 = y - int(d/4)
                if x1<0:
                    x1 = 0
                if y1<0:
                    y1 = 0
                y2 = int(y1+1.5*d)
                x2 = int(x1+1.5*d)
                image_shape = frame.shape[:2]
                h, w = image_shape
                if y2>h:
                    y2 = h
                if x2 > w:
                    x2=w
                #frame= cv2.rectangle(frame, (x1, y1), (int(x1+1.5*d), int(y1+1.5*d)), (0, 0, 200),1)
                self.bboxes.append([x1, y1, x2, y2])

            self._reset()
            self.previus_frame = gray
            if (len(self.bboxes)>0):
                return True
        return False

    