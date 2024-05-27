class Detection:
    def __init__(self, ai_box, movement_box, confirmed_box) -> None:
        self.ai_box = ai_box
        self.movement_box = movement_box
        self.confirmed_box = confirmed_box
        self.delete_me = False
    
    

    def copy(self):
        return Detection(self.ai_box,self.movement_box,self.confirmed_box)