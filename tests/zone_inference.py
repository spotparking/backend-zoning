# from modules import Zone #TODO: implement the Zone class so that we can use the zone the right way
from pathlib import Path
import numpy as np
import random
import cv2
import os

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
    for folder in zone:
        enter_hists[folder] = average_color_histogram(path_to_data + "/" + folder + "/enter")
        leave_hists[folder] = average_color_histogram(path_to_data + "/" + folder + "/leave")
    
    # for each exit, find the best match from the enters
    correct_matches = 0
    incorrect_matches = 0
    for leave_car in leave_hists.keys():
        best_match = None
        best_match_score = 0
        for enter_car in enter_hists.keys():
            score = compare_histograms(leave_hists[leave_car], enter_hists[enter_car])
            if score > best_match_score:
                best_match_score = score
                best_match = enter_car
        if best_match == leave_car:
            correct_matches += 1
        else:
            incorrect_matches += 1

    # correct_matches + incorrect_matches should equal the zone_size
    return correct_matches, incorrect_matches

def average_color_histogram(folder_path):
    folder_path = Path(folder_path)
    hist_sum = None
    count = 0

    for image_path in folder_path.glob("*.jpg"):
        image = cv2.imread(str(image_path))

        # Convert image to HSV for better color representation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        hist_s = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

        # Concatenate histograms into one
        hist = np.concatenate((hist_h, hist_s, hist_v), axis=0)

        # Add to sum
        if hist_sum is None:
            hist_sum = hist
        else:
            hist_sum += hist
        count += 1

    if count == 0:
        return None

    # Calculate the average
    avg_hist = hist_sum / count
    return avg_hist

def compare_histograms(hist1, hist2):
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)