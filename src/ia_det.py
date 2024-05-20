import time
import cv2
from openvino.runtime import Core
import numpy as np

class IaDetector:
    def __init__(self,confidence_thr=0.5) -> None:
        self.confidence_thr=confidence_thr
        core = Core()
        core.set_property({'CACHE_DIR': './openvino_cache'})
        model = core.read_model(model='model/pd0200.xml')
        self.compiled_model = core.compile_model(model=model)
        self.bboxes = []
    
    def _get_images(self,frame,bboxes):
        images = []
        for b in bboxes:
            # crop_img = img[y:y+h, x:x+w]
            x, y, x_max, y_max = b
            img = frame[y:y_max,x:x_max].copy()
            images.append([img,x,y,x_max,y_max])
        return images
    
    def _resize_images(self,images):
        res_images = []
        for image in images:
            img, x, y, x_max, y_max = image
            img = cv2.resize(img, dsize=[256,256])
            #cv2.imshow('img', img)
            img = np.expand_dims(img.transpose(2,0,1), axis=0)
            res_images.append([img, x, y, x_max, y_max])
        return res_images
        
    
    def process_bboxes(self,frame,bboxes):
        images_with_bbox = self._get_images(frame,bboxes)
        input_images = self._resize_images(images_with_bbox)

        pred_images = []
        for image in input_images:
            img, x, y, x_max, y_max = image
            pred_scores = self.compiled_model( [img] )
            m200 = pred_scores[0][0][0]
            filtered = m200[m200[:,2] > self.confidence_thr]
            if (len(filtered)>0):
                pred_images.append([img,filtered,x, y, x_max, y_max])
        self.bboxes = pred_images
        if len(pred_images)>0:
            return True
        return False