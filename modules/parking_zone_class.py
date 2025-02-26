
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

    def average_color_histogram(frames):
        
        hist_sum = None
        count = 0

        for image in frames:
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
    
    def get_drive_pics(self, frames:list[np.ndarray], record:pd.DataFrame, frame_skip=1, track_id_col='track_id'):
        # TODO: TEST THIS
        
        record = record.copy()
        if 'in_driving_region' not in record.columns:
            record['in_driving_region'] = is_in_zone_vec(record[['cx', 'cy']].values.astype(int), self.driving_region_pix)
        
        frame_tracks = {}
        record['br_x'] = record['tl_x'] + record['w']
        record['br_y'] = record['tl_y'] + record['h']
        for iteration, frame in record.groupby('iteration'):
            frame = frame.loc[frame['confirmed'], :]
            bbox_df = frame[[track_id_col, 'confidence', 'tl_x', 'tl_y', 'br_x', 'br_y']]
            bbox_arr = bbox_df.to_numpy().astype(float)
            frame_tracks[int(iteration)-1] = bbox_arr
            
        drive_pics = []
        for i, frame in enumerate(frames):
            if i % frame_skip != 0:
                continue
            bbox = frame_tracks.get(i, None)
            if bbox is not None:
                _, _, tl_x, tl_y, br_x, br_y = bbox
                crop = frame[int(tl_y):int(br_y), int(tl_x):int(br_x)]
                drive_pics.append(crop)
            
        return drive_pics
    
    def create_feature(self, frames:list[np.ndarray], record:pd.DataFrame, zone_driving_region_coordinates:list[list[float]], license_plate:str) -> np.ndarray:
        # TODO: TEST THIS
        
        drive_pics = self.get_drive_pics(frames, record)
        
        ave_hist = ParkingZone.average_color_histogram(drive_pics)
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
        closest_k = ParkingZone.closest_k_points(leaving_car.get_center_pt(), [car.get_center_pt() for car in self.cars], k=k)
        
        for enter_car in self.cars:
            if enter_car.get_center_pt() in closest_k:
                score = ParkingZone.get_vector_similarity(leaving_car.get_ave_hist(), enter_car.get_ave_hist())
            if score > best_match_score:
                    best_match_score = score
                    best_match = enter_car
            
        return best_match