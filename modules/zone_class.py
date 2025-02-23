from modules.car_class import Car
from modules.helpers import closest_k_points, compare_histograms

class Zone:
    def __init__(self, zone_id:str, cam_label:str, coordinates:list[list[float]], driving_region_coordinates:list[list[float]]):
        self.zone_id = zone_id
        self.cam_label = cam_label
        self.coordinates = coordinates
        self.driving_region_coordinates = driving_region_coordinates
        
        self.cars = []
    
    
    # not sure if these are needed with the current implementation vvv
    def load_cars(self):
        # getting the current cars from memory when a new car enters or leaves
        pass
    
    def save_cars(self):
        # saving the current cars to memory
        pass
    # not sure if these are needed with the current implementation ^^^
    
    
    def add_car(self, car:Car):
        self.cars.append(car)
    
    def remove_car(self, car):
        self.cars.remove(car)
    
    def predict(self, leaving_car, k): # could remove k and just standardize for each zone with a specific k val
        best_match = None
        best_match_score = 0
        
        if len(self.cars) < k:
            k = len(self.cars)
        closest_k = closest_k_points(leaving_car.center_pt, [car.center_pt for car in self.cars], k=k)
        
        for enter_car in self.cars:
            if enter_car.center_pt in closest_k:
                score = compare_histograms(leaving_car.ave_hist, enter_car.ave_hist)
            if score > best_match_score:
                    best_match_score = score
                    best_match = enter_car
            
        return best_match