import collections

class PathHandler:
    def __init__(self, positives_required) -> None:
        self.positives_required = positives_required
        self.paths = []

    def process_paths(self, frame_detections):
        # if there is no detections in the current frame, we delete all paths tracked
        if len(frame_detections) == 0:
            self.paths = []
            return
        for path in self.paths:
            res = path.next_step(frame_detections)
            if res:
                frame_detections = list(filter(lambda x: x.delete_me == False, frame_detections))
        self.paths = list(filter(lambda x: x.delete_me == False, self.paths))

        # start new paths with the rest of detections
        for det in frame_detections:
            self.paths.append(Path(det))
    
    def get_positives(self):
        boxes = []
        trackpoints = []
        for path in self.paths:
            alarm, len_, last, trackpt = path.status(self.positives_required)
            if alarm:
                x1, y1, x2, y2 = last.confirmed_box.get_xyxy()
                boxes.append([x1,y1,x2,y2,len_])
                trackpoints.append(trackpt)
        return boxes,trackpoints


class Path:
    def __init__(self, first_detection) -> None:
        self.path = collections.deque()
        self.path.append(first_detection)
        self.delete_me = False
    
    def next_step(self,list_of_detections):
        previous_det = self.path[0]
        accepted_det = None
        accepted_coverage=0
        for det in list_of_detections:
            confirmed = det.confirmed_box
            coverage, _ = previous_det.confirmed_box.coverage(confirmed)
            if coverage>accepted_coverage:
                accepted_coverage = coverage
                accepted_det = det
        if accepted_coverage>0:
            self.path.appendleft(accepted_det.copy())
            accepted_det.delete_me = True
            return True

        else:
            self.delete_me = True
        return False
    
    def get_track_points(self):
        # return x_max+x_min / 2, y_max or something like that
        points = []
        for det in self.path:
            points.append(det.confirmed_box.get_trackpoint())
        return points
    
    def status(self, positives_required):
        len_path = len(self.path)
        if len_path>=positives_required:
            # alarm, len, last_detection, track_points
            return True,len_path, self.path[0], self.get_track_points()
        
        return False, len_path, None, None

