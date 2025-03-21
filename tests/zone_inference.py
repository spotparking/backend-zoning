# from modules import Zone #TODO: implement the Zone class so that we can use the zone the right way
import random
import os
import pandas as pd
import numpy as np
import ast

from modules.spot_classes.car_class import Car
from modules.parking_zone_class import ParkingZone
from modules.helpers import get_images_from_folder

def test_zone_inference(zone_size:int, path_to_data:str):
    
    ###############################################################################################################################
    # path_to_data: path to the folder containing the data (normally named after a lot or camera)                                 #
    #       path_to_data is treated like it has a subfolder for each car which then has a subfolder for enter and leave pictures  #
    #       ex: path_to_data/{license_plate}/enter/{image}.jpg                                                                    #
    ###############################################################################################################################
    
    CRABTREE_PARKING_REGION_PIX = np.array([
        [545, 129],
        [674, 156],
        [774, 291],
        [610, 263],
    ])
    CRABTREE_DRIVING_REGION_PIX = np.array([
        [357, 60],
        [522, 85],
        [666, 385],
        [401, 385],
    ])
    zone = ParkingZone("zone1", "cam1", CRABTREE_PARKING_REGION_PIX, CRABTREE_DRIVING_REGION_PIX)
    
    data_folders = [f for f in os.listdir(path_to_data) if os.path.isdir(os.path.join(path_to_data, f))]
    cars_to_add = random.sample(data_folders, zone_size)
    
    all_cars = pd.read_csv(path_to_data + "all_cars.csv")
    leaving_cars:list[Car] = []
    matches = {}
    for folder in cars_to_add:
        
        entering_car = Car()
        enter_frames = get_images_from_folder(path_to_data + "/" + folder + "/enter")
        ave_hist = ParkingZone.average_color_histogram(enter_frames)
        cx, cy = tuple(int(x) for x in ast.literal_eval(all_cars[all_cars["license_plate"] == folder]["last_enter_pt"].values[0]))
        enter_feature = np.concatenate((ave_hist, np.array([cx, cy])))
        entering_car.set_feature(enter_feature)
        zone.add_car(entering_car)
        
        leaving_car = Car()
        leave_frames = get_images_from_folder(path_to_data + "/" + folder + "/leave")
        ave_hist = ParkingZone.average_color_histogram(leave_frames)
        cx, cy = tuple(int(x) for x in ast.literal_eval(all_cars[all_cars["license_plate"] == folder]["first_leave_pt"].values[0]))
        leave_feature = np.concatenate((ave_hist, np.array([cx, cy])))
        leaving_car.set_feature(leave_feature)
        leaving_cars.append(leaving_car)
        
        # TODO: Change this to use the ParkingZone.add_car function
        
        matches[entering_car] = leaving_car
    
    correct_matches = 0
    incorrect_matches = 0
    for leaving_car in leaving_cars:
        best_match = None
        best_match_score = 0
        
        # TODO: Change this to use the ParkingZone.add_car and ParkingZone.get_match functions
        
        # find closest k enter pts
        enter_pts = [car.get_center_pt() for car in zone.cars]
        closest_k = ParkingZone.closest_k_points(leaving_car.get_center_pt(), enter_pts, 2)
        
        for enter_car in zone.cars:
            if enter_car.get_center_pt() in closest_k:
                score = ParkingZone.get_vector_similarity(leaving_car.get_ave_hist(), enter_car.get_ave_hist())
                if score > best_match_score:
                    best_match_score = score
                    best_match = enter_car

        if best_match is not None:
            if matches[best_match] == leaving_car:
                correct_matches += 1
            else:
                incorrect_matches += 1

    return correct_matches, incorrect_matches