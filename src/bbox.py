class Bbox:
    def __init__(self, x1, y1, x2, y2, label: str = "Undefined_class", confidence: float = 0.0):
        self.x_max = max(x1, x2)
        self.x_min = min(x1, x2)
        self.y_max = max(y1, y2)
        self.y_min = min(y1, y2)
        self.label = label

        if confidence > 1:
            self.confidence = confidence / 100
        else:
            self.confidence = confidence

    @property
    def area(self) -> int:
        """
        Calculates the surface area. useful for IOU!
        """
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def intersect(self, bbox) -> int:
        if bbox.label != self.label:
            return 0

        x1 = min(self.x_max, bbox.x_max)
        y1 = min(self.y_max, bbox.y_max)
        x2 = max(self.x_min, bbox.x_min)
        y2 = max(self.y_min, bbox.y_min)

        intersection = max(0, x1 - x2) * max(0, y1 - y2)

        return intersection, Bbox(x1,y1,x2,y2)#[min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)]

    def iou(self, bbox) -> float:
        intersection, box = self.intersect(bbox)

        iou = intersection / float(self.area + bbox.area - intersection)
        # return the intersection over union value
        return iou

    def coverage(self,bbox):
        intersection, box = self.intersect(bbox)
        cov =  intersection/self.area
        return cov, box
    
    def get_xyxy(self):
        return [self.x_min,self.y_min,self.x_max,self.y_max]
    
    def get_trackpoint(self):
        return [int((self.x_min + self.x_max)/2),self.y_max]