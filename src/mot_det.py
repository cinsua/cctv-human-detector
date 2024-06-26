import cv2
from src.bbox import Bbox

class MotionDetector:
    def __init__(self,frame, n_frames = 10, treshold = 25) -> None:
        self.bboxes = []
        self.n_frames = n_frames
        self.treshold =treshold
        self._reset()
        height_camera, width_camera, _ = frame.shape

        # adjust the kernels based on the resolution of the camera
        k_e = int(height_camera/100)
        w_k = int(height_camera/20)
        h_k = int(w_k*2)
        
        # compensation for reducing the rectangles in each side
        # the kernels transformation are used to join small detections but gives rectangles more bigger than should be
        self.compensation = int(w_k/2)

        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_e, k_e))
        self.kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w_k, w_k))
        self.kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (w_k, h_k))

        self.min_contour_area = w_k * h_k * 2
    
    def _reset(self):
        self.frame_counter = 0
    
    def _preprocess(self,frame):
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #f = cv2.GaussianBlur(f, (21,21), 0)
        return f
    
    def process_frame(self,frame):
        self.frame_counter+=1
        
        # 5 frames earlier as a background is fine, maybe it should be a parameter
        # N_FRAMES should be bigger than 5
        if (self.frame_counter%self.n_frames == self.n_frames-5):
            self.previus_frame = self._preprocess(frame)

        if (self.frame_counter%self.n_frames == 0):
            gray = self._preprocess(frame)
            delta = cv2.absdiff(gray, self.previus_frame)
            # Reduce Noise
            delta_thresh = cv2.threshold(delta, self.treshold, 255, cv2.THRESH_BINARY)[1]
            delta_thresh = cv2.erode(delta_thresh, self.kernel_erode, iterations=1)
            # Expand the movement detected to get a better zone
            delta_thresh_dilated = cv2.dilate(delta_thresh, self.kernel_rect, iterations=1)
            delta_thresh_dilated = cv2.dilate(delta_thresh_dilated, self.kernel_ellipse, iterations=1)
            # get countours, filter the small and isolated zones that erode couldnt remove
            contours, hierarchy = cv2.findContours(delta_thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]

            # reduce the sides to compensate the morphs
            # checking if this is really helping
            bboxes = []
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x = x + self.compensation
                y = y + self.compensation
                w = w - self.compensation*2
                h = h - self.compensation*2
                bboxes.append(Bbox(x,y,x+w,y+h))

            self.bboxes = bboxes
            self._reset()
            # if we get movement, return True
            if (len(self.bboxes)>0):
                return True
        return False

    