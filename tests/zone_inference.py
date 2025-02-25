# from modules import Zone #TODO: implement the Zone class so that we can use the zone the right way
import random
import os
import pandas as pd
import numpy as np
import ast

from modules.parking_zone_class import ParkingZone

def test_zone_inference(zone_size:int, path_to_data:str):
    
    ###############################################################################################################################
    # path_to_data: path to the folder containing the data (normally named after a lot or camera)                                 #
    #       path_to_data is treated like it has a subfolder for each car which then has a subfolder for enter and leave pictures  #
    #       ex: path_to_data/{license_plate}/enter/{image}.jpg                                                                    #
    ###############################################################################################################################
    
    # zone = Zone() #TODO: implement the Zone class so that we can use the zone the right way
    
    data_folders = [f for f in os.listdir(path_to_data) if os.path.isdir(os.path.join(path_to_data, f))]
    zone = random.sample(data_folders, zone_size)
    
    # create average color histogram for each car's enter and leave
    enter_hists = {}
    leave_hists = {}
    last_enter_pt = {}
    first_leave_pt = {}
    all_cars = pd.read_csv(path_to_data + "all_cars.csv")
    for folder in zone:
        enter_hists[folder] = ParkingZone.average_color_histogram(path_to_data + "/" + folder + "/enter")
        leave_hists[folder] = ParkingZone.average_color_histogram(path_to_data + "/" + folder + "/leave")
        last_enter_pt[folder] = tuple(int(x) for x in ast.literal_eval(all_cars[all_cars["license_plate"] == folder]["last_enter_pt"].values[0]))
        first_leave_pt[folder] = tuple(int(x) for x in ast.literal_eval(all_cars[all_cars["license_plate"] == folder]["last_enter_pt"].values[0]))
    
    # for each exit, find the best match from the enters
    correct_matches = 0
    incorrect_matches = 0
    for leave_car in leave_hists.keys():
        best_match = None
        best_match_score = 0
        
        leave_pt = last_enter_pt[leave_car]
        # find closest k enter pts
        closest_k = ParkingZone.closest_k_points(leave_pt, first_leave_pt.values(), 1)
        
        for enter_car in enter_hists.keys():
            if last_enter_pt[enter_car] in closest_k:
                score = ParkingZone.get_vector_similarity(leave_hists[leave_car], enter_hists[enter_car])
                if score > best_match_score:
                    best_match_score = score
                    best_match = enter_car
        if best_match == leave_car:
            correct_matches += 1
        else:
            incorrect_matches += 1

    return correct_matches, incorrect_matches