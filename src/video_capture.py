import cv2
import time

class VideoCapture:
    def __init__(self,width, height, fps,output_dir = 'videos/') -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.output_dir = output_dir
        #self.fourcc = cv2.VideoWriter_fourcc(*'MP42') # .avi
        self.create_new_file = True
        self.filename = ''
        self.n_frames = 0
        self.recording = False

        # in windows64 we should install codecs for this. test required
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1') #
        #self.fourcc = 0x31637661
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V') # .avi
        #self.fourcc = cv2.VideoWriter_fourcc(*'MP42') # .avi. in MP4 gives an error, but works fine, and compatible with wsp
        #self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') # .mp4 NO COMPATIBLE WITH WHATSAPP
        #self.fourcc = cv2.VideoWriter_fourcc(*'AVC1') # in MP4 gives an error, but works fine, and compatible with wsp
        #self.fourcc = cv2.VideoWriter_fourcc(*'H264') # error libx264
        #self.fourcc = cv2.VideoWriter_fourcc(*'avc1') # MP4 gives an error, but works fine, and compatible with wsp
        #fourcc = cv2.VideoWriter_fourcc(*'WRAW') # error --- no information ---
        #self.fourcc = cv2.VideoWriter_fourcc(*'MPEG') # .avi 30fps
        #self.fourcc = cv2.VideoWriter_fourcc(*'MJPG') # .avi
        #self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi
        #self.fourcc = cv2.VideoWriter_fourcc(*'H265') # error
        '''
        si pones -1 en fourcc tira este listado (?)
        fourcc tag 0x7634706d/'mp4v' codec_id 000C  2.46 not compatible wsp
        fourcc tag 0x31637661/'avc1' codec_id 001B  error but compatible
        fourcc tag 0x33637661/'avc3' codec_id 001B  error but compatible. freezes one sec
        fourcc tag 0x31766568/'hev1' codec_id 00AD  5.29 error, freezes
        fourcc tag 0x31637668/'hvc1' codec_id 00AD  err
        fourcc tag 0x7634706d/'mp4v' codec_id 0002  2.45 not compatible wsp
        fourcc tag 0x7634706d/'mp4v' codec_id 0001  
        fourcc tag 0x7634706d/'mp4v' codec_id 0007
        fourcc tag 0x7634706d/'mp4v' codec_id 003D
        fourcc tag 0x7634706d/'mp4v' codec_id 0058
        fourcc tag 0x312d6376/'vc-1' codec_id 0046  5.29 etc
        fourcc tag 0x63617264/'drac' codec_id 0074  10.4mb no error NO compatible
        fourcc tag 0x7634706d/'mp4v' codec_id 00A3
        fourcc tag 0x39307076/'vp09' codec_id 00A7  small. no error but slows down everything
        fourcc tag 0x31307661/'av01' codec_id 801D  small. no error but slows down everything
        fourcc tag 0x6134706d/'mp4a' codec_id 15002 5.29 etc
        
        '''
    
    def process_frame(self,frame_buffer):
        if self.n_frames == 0:
            if self.recording:
                self.fout.release()
                print(f'Terminado de grabar en {self.filename}')
                self.recording = False
            self.create_new_file=True
            return
        
        self.n_frames -=1

        if self.create_new_file:
            
            self.recording = True
            self.filename = time.strftime(f"{self.output_dir}%Y.%m.%d  %H.%M.%S", time.localtime()) + ".mp4"
            self.fout = cv2.VideoWriter(self.filename, self.fourcc, self.fps, (self.width, self.height))
            #self.fout = cv2.VideoWriter(self.filename, -1, self.fps, (self.width, self.height))
            self.create_new_file = False
            if not self.fout.isOpened():
                print("ERROR INITIALIZING VIDEO WRITER")
                #break
            else:
                print("OK INITIALIZING VIDEO WRITER")
            
            print(f'Comenzando a grabar en {self.filename}')
        if self.fout.isOpened():
            frame_out = frame_buffer.popleft()
            height_f, width_f, _ = frame_out.shape
            if (self.width!=width_f) or (self.height!=height_f):
                frame_out = cv2.resize(frame_out, (self.width, self.height))
            self.fout.write(frame_out)
            
            #fout.release() # close current file
            #create_new_file = True # time to create new file in next loop
    
    def start_recording(self,n_frames=30):
        self.n_frames = n_frames
