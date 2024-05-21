import time
class Fps:
    def __init__(self, n_frames=10) -> None:
        
        self.n_frames = n_frames
        self.avg_fps = 0
        self.total_fps = 0
        self._reset()
    
    def _reset(self):
        self.total_time = 0
        self.frame_counter = 0

    
    def start_fps(self):
        t = time.perf_counter()
        self.start_time = t
    
    def end_fps(self):
        t = time.perf_counter()
        time_elapsed = t - self.start_time
        self.total_fps += 1.0/time_elapsed

        self.total_time += time_elapsed
        self.frame_counter +=1
        if (self.frame_counter%self.n_frames == 0):
            #avg_time = self.total_time / self.n_frames
            self.avg_fps = self.total_fps / self.frame_counter
            #self._reset()
    
    def get_fps_label(self):
        return f'FPS {self.avg_fps:.1f}'
            

    
