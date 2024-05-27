import time
import cv2
from openvino.runtime import Core
import numpy as np
from src.bbox import Bbox

class IaDetector:
    def __init__(self,confidence_thr=0.8) -> None:
        self.confidence_thr=confidence_thr
        core = Core()
        core.set_property({'CACHE_DIR': './openvino_cache'})
        model = core.read_model(model='model/FP16-instance-segmentation-person-0007.xml')
        self.compiled_model = core.compile_model(model=model)
        self.bboxes = []
        self.scores = []

    def reset_results(self):
        self.bboxes = []
        self.scores = []


    def _rezise_frame(self,frame):
        input_frame = cv2.resize(frame, dsize=[544,320])
        input_frame = np.expand_dims(input_frame.transpose(2,0,1), axis=0)
        return input_frame
    
    def process_frame(self,frame):
        input_frame = self._rezise_frame(frame)
        pred_scores = self.compiled_model( [input_frame] )
        res = pred_scores[0]
        filtered = res[res[:,4] > self.confidence_thr]
        image_shape = frame.shape[:2]
        h, w = image_shape
        h=h/320
        w=w/544

        scores = []
        bboxes = []

        for f in filtered:
            score = int(f[4] *100)
            x_min = int(w*f[0])
            y_min = int(h*f[1])
            x_max = int(w*f[2])
            y_max = int(h*f[3])
            scores.append(f"IA {score}%")
            bboxes.append(Bbox(x_min, y_min, x_max, y_max))
        self.scores = scores
        self.bboxes = bboxes
        if self.bboxes ==[]:
            return False
        return True