import random


class Zone:
    
    def __init__(self, zone_id:str, cam_label:str, coordinates:list[list[float]]):
        
        self.cars = []
        pass
    
    def load_cars(self):
        # getting the current cars from memory when a new car enters or leaves
        pass
    
    def save_cars(self):
        # saving the current cars to memory
        pass
    
    def add_car(self, car):
        pass
    
    def remove_car(self, car):
        pass
    
    def predict(self, car_vector):
        # return the car from self.cars with the best match to car_vector
        return random.choice(self.cars)
    
    