
from typing import Tuple
import random

import numpy as np
import pandas as pd
import cv2

from modules.car_class import Car
from modules.helpers import is_in_zone_vec, points_float_to_pix
import modules.data_manager as dm

class ParkingZone:
    def __init__(self, zone_id:str, cam_label:str, coordinates:list[list[float]], driving_region_coordinates:list[list[float]]=None):
        self.zone_id:str = zone_id
        self.cam_label:str = cam_label
        self.coordinates:list[list[float]] = coordinates
        if driving_region_coordinates is None:
            driving_region_coordinates = []
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
        if not isinstance(car, Car):
            raise TypeError(f"car must be a Car object, got {type(car)}")
        self.cars.append(car)
    
    
    def remove_car(self, car):
        self.cars.remove(car)
        
    def clear_cars(self):
        self.cars:list[Car] = []
        
    # START DEV FUNCTION
    def shuffle_cars(self):
        random.shuffle(self.cars)
    # END DEV FUNCTION
        
    
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
        closest_points = [point for point, _ in sorted(distances, key=lambda x: x[1])[:k]]
        # the get_center_pt function returns (0, 0) if it can't find the point, so these MUST be included
        # since the distance metric does not work on these instances
        zero_points = [point for point, dist in distances if point[0] == 0 and point[1] == 0]
        return set(closest_points + zero_points)


    def get_vector_similarity(hist1:np.ndarray, hist2:np.ndarray) -> float:
        hist1 = hist1.astype(np.float32)
        hist2 = hist2.astype(np.float32)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    
    def average_color_histogram(frames, BIN_SIZE=8):
        
        hist_sum = None
        count = 0

        for image in frames:
            # Convert image to HSV for better color representation
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histogram for each channel
            hist_h = cv2.calcHist([hsv_image], [0], None, [BIN_SIZE], [0, 256]).reshape(-1)
            hist_s = cv2.calcHist([hsv_image], [1], None, [BIN_SIZE], [0, 256]).reshape(-1)
            hist_v = cv2.calcHist([hsv_image], [2], None, [BIN_SIZE], [0, 256]).reshape(-1)

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


    def get_center_pt(self, record: pd.DataFrame, action='enter', cx_col='cx_ma', cy_col='cy_ma') -> Tuple[int, int]:
        """
        Get the center point (cx, cy) from the given record DataFrame based on the specified condition.
        Parameters:
        record (pd.DataFrame): The DataFrame containing the records with 'cx' and 'cy' columns.
        action (str): The action to perform. Must be one of ['enter', 'leave']. Default is 'enter'.
        
        Returns:
        Tuple[int, int]: The center point (cx, cy) as a tuple of integers.
        Raises:
        ValueError: If the 'which' parameter is not one of the acceptable values.
        """
        # add the in_driving_region column if it doesn't exist
        record = record.copy()
        if 'in_driving_region' not in record.columns:
            record['in_driving_region'] = self.compute_in_driving_region(record, cx_col=cx_col, cy_col=cy_col)
        
        curr_in_driving_region = (record['in_driving_region'].values).astype(bool)
        next_in_driving_region = (record['in_driving_region'].shift(-1).values == True).astype(bool)
        next_in_driving_region[-1] = curr_in_driving_region[-1]
        
        # if we are trying to find the leave point, then we want the first time that
        # the car went from in_driving_region -> not in_driving_region
        if action.lower() == 'leave':
            switches_mask = ~curr_in_driving_region & next_in_driving_region
            if any(switches_mask):
                index = record.index[switches_mask].values[0]
            # WARNING: THIS FORCES THE CAR TO LEAVE ON THE FIRST FRAME
            else:
                index = None
                # print("WARNING: Using first in_driving_point point as leave point")
                switches_mask = ~curr_in_driving_region
                if any(switches_mask):
                    index = record.index[switches_mask].values[0]
                else:
                    # print("WARNING: Using first point as leave point")
                    index = record.index.values[0]
        # if we are trying to find the enter point, then we want the last time that
        # the car went from not in_driving_region -> in_driving_region
        elif action.lower() == 'enter':
            switches_mask = curr_in_driving_region & ~next_in_driving_region
            if any(switches_mask):
                index = record.index[switches_mask].values[-1]
            # WARNING: THIS FORCES THE CAR TO ENTER ON THE LAST FRAME
            else:
                index = None
                # print("WARNING: Using last in_driving_region point as enter point")
                switches_mask = curr_in_driving_region
                if any(switches_mask):
                    # print("WARNING: Using last point as enter point")
                    index = record.index[switches_mask].values[-1]
                else:
                    index = record.index.values[-1]
        # if the user didn't specify action correctly then raise an error
        else:
            raise ValueError(f"action='{action}' not in ['enter', 'leave']")
        
        # if it could not find the center point, just return (0,0)
        if index is None:
            return (0, 0)
            # raise ValueError(f"Could not find {action} point")
        
        # get the center point from the record as a tuple of integers
        center_pt = record.loc[index, [cx_col, cy_col]]
        return tuple(map(int, center_pt))
    
    
    def get_drive_pics(self, frames:list[np.ndarray], record:pd.DataFrame, frame_skip=1, track_id:int|str=None, track_id_col='track_id'):
        
        record = record.copy()
        if 'in_driving_region' not in record.columns:
            record['in_driving_region'] = is_in_zone_vec(record[['cx', 'cy']].values.astype(int), self.driving_region_pix)
        
        # check the track_id_col to make sure it is in the record
        if track_id_col not in record.columns:
            raise ValueError(f"track_id_col='{track_id_col}' not in record.columns")
        
        # filter the record to only be of the track_id
        if track_id is not None:
            record = record.loc[record[track_id_col] == track_id, :]
            if len(record) == 0:
                raise ValueError(f"track_id={track_id} not found in record")
        
        # map each iteration to the bounding boxes of the cars in that iteration
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
                _, _, tl_x, tl_y, br_x, br_y = bbox[0]
                crop = frame[int(tl_y):int(br_y), int(tl_x):int(br_x)]
                drive_pics.append(crop)
            
        return drive_pics
    
    
    def compute_in_driving_region(self, record:pd.DataFrame, cx_col='cx_ma', cy_col='cy_ma') -> pd.DataFrame:
        return ~is_in_zone_vec(
            record[[cx_col, cy_col]].values.astype(int), 
            np.array(self.coordinates).astype(int)
        )
    
    
    def create_feature(self, frames:list[np.ndarray], record:pd.DataFrame, action) -> np.ndarray:
        
        if action.lower() != 'enter' and action.lower() != 'leave':
            raise ValueError(f"action='{action}' not in ['enter', 'leave']")
        
        drive_pics = self.get_drive_pics(frames, record)

        ave_hist = ParkingZone.average_color_histogram(drive_pics)
        
        center_pt = self.get_center_pt(record, action=action)
        
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



    
    
############################################################
#               PARKINGZONE CREATION FUNCTION              #
############################################################
    
    
    


def get_parking_zone_from_zoneID(zoneID:str) -> ParkingZone:
    # load the zone_settings based on the zoneID
    cam_label, region_id, zone_name = dm.parse_zoneID(zoneID)
    settings = dm.get_settings(cam_label)
    zone_settings = dm.get_zone_settings(zoneID, settings=settings)
    
    # load the coordinates as pixels instead of floats
    resolution = dm.get_camera_resolution(cam_label, settings=settings)
    height = resolution['height']
    width = resolution['width']
    coordinates = points_float_to_pix(zone_settings['points'], height, width)
    
    # create and return the ParkingZone object
    return ParkingZone(zoneID, cam_label, coordinates, [])