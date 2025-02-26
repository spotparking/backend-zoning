
import numpy as np
import pandas as pd
import cv2

from car_class import Car
from modules.helpers import is_in_zone_vec

class ParkingZone:
    def __init__(self, zone_id:str, cam_label:str, coordinates:list[list[float]], driving_region_coordinates:list[list[float]]):
        self.zone_id:str = zone_id
        self.cam_label:str = cam_label
        self.coordinates:list[list[float]] = coordinates
        self.driving_region_coordinates:list[list[float]] = driving_region_coordinates
        
        self.cars:list[Car] = []
        
        ParkingZone.zones_registry[zone_id] = self
    
    ############################################################
    #                      OBJECT METHODS                      #
    ############################################################
    
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
        
    ############################################################
    #                    FEATURE VECTOR CODE                   #
    ############################################################
    
    def closest_k_points(leave, enter_points, k):
        """
        Used to find the closest k entering instances compared to the leaving instance
        """
        # Compute distances and store them with points
        distances = [(point, np.linalg.norm(np.array(point) - np.array(leave))) for point in enter_points]
        
        # Sort by distance and return the k closest points
        return [point for point, _ in sorted(distances, key=lambda x: x[1])[:k]]

    def get_vector_similarity(hist1:np.ndarray, hist2:np.ndarray) -> float:
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def take_drive_pics(record:pd.DataFrame, frames:list[np.ndarray], driving_region_pix:list[list[float]]) -> list[np.ndarray]:
        record = record.copy()
        record['in_driving_region'] = is_in_zone_vec(record[['cx', 'cy']].values.astype(int), driving_region_pix)
        
        cropped_images = []
        for i, row in record.iterrows():
            if row['in_driving_region']:
                frame = frames[row['iteration']]
                if not np.isnan(row["tl_y"]) and not np.isnan(row["tl_x"]) and not np.isnan(row["h"]) and not np.isnan(row["w"]):
                    cropped = frame[int(row["tl_y"]):(int(row["tl_y"]) + int(row["h"])),
                                    int(row["tl_x"]):(int(row["tl_x"]) + int(row["w"]))]
                    cropped_images.append(cropped)
        return cropped_images

    def average_color_histogram(record, frames, driving_region_coordinates):
        cropped_images = ParkingZone.take_drive_pics(record, frames, driving_region_coordinates)
        
        hist_sum = None
        count = 0

        for image in cropped_images:
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
        
        avg_hist = hist_sum / count
        return avg_hist

    def get_center_pt(record, driving_region_pix):
        record = record.copy()
        record['in_driving_region'] = is_in_zone_vec(record[['cx', 'cy']].values.astype(int), driving_region_pix)
        
        last_center_pt = None
        for i, row in record.iterrows():
            if row['in_driving_region']:
                last_center_pt = row[["cx", "cy"]]
        return last_center_pt
    
    def create_feature(frames:list[np.ndarray], record:pd.DataFrame, zone_driving_region_coordinates:list[list[float]], license_plate:str) -> np.ndarray:
        
        ave_hist = ParkingZone.average_color_histogram(record, frames, zone_driving_region_coordinates)
        center_pt = ParkingZone.get_center_pt(record, zone_driving_region_coordinates)
        
        # Other future features (to be saved in the car class):
            # average width height at each driving picture
            # saved cropped images for classifier that takes 2 images and says if they have the same car in them
        
        return np.concatenate([ave_hist, center_pt])
    
    ############################################################
    #                        PREDICTION                        #
    ############################################################

    def get_match(self, leaving_car:Car, k:int) -> Car: # could remove k and just standardize for each zone with a specific k val
        best_match = None
        best_match_score = 0
        
        if len(self.cars) < k:
            k = len(self.cars)
        closest_k = ParkingZone.closest_k_points(leaving_car.center_pt, [car.center_pt for car in self.cars], k=k)
        
        for enter_car in self.cars:
            if enter_car.center_pt in closest_k:
                score = ParkingZone.get_vector_similarity(leaving_car.ave_hist, enter_car.ave_hist)
            if score > best_match_score:
                    best_match_score = score
                    best_match = enter_car
            
        return best_match