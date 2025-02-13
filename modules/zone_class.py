
class Zone:
    
    def __init__(self, zone_id:str, cam_label:str, coordinates:list[list[float]]):
        
        self.cars = []
        pass
    
    def load_cars(self):
        pass
    
    def save_cars(self):
        pass
    
    def add_car(self, car):
        pass
    
    def remove_car(self, car):
        pass
    
    def predict(self, car_vector):
        # return the car from self.cars with the best match to car_vector
        pass
    
    