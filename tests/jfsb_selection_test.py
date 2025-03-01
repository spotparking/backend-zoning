if __name__ == "__main__":
    ############################################################
    #                       SCRIPT SETUP                       #
    ############################################################

    import sys
    import os

    sys.path.append(os.getcwd())

    # if running this script from the scripts directory, then move up one directory
    # so that the script runs from the proficiency_assessment directory
    if 'tests' in os.getcwd():
        os.chdir("..")
        sys.path.append(os.getcwd())
        print("Changed directory to", os.getcwd())
    if 'backend-zoning' not in os.getcwd():
        print("Please run this script from the SpotBackend directory.")
        sys.exit()

    ############################################################
    #                           MAIN                           #
    ############################################################

from copy import deepcopy
import random
from itertools import combinations
from typing import Dict

import pandas as pd
from tqdm import tqdm
from jeffutils.utils import stack_trace

import modules.data_manager as dm
from modules.parking_zone_class import ParkingZone
from modules.display import show_video
from modules.spot_classes.car_class import Car
from tests.test_helpers import (
    load_test_set, 
    get_enter_videos, 
    get_leave_videos, 
    get_video_pairs, 
)

def test_parking_zone_inference(zone_size:int, zoneID:str, test_set_label:str, max_tests:int=100, k=5, cars_record:Dict[str, Car]={}):
    
    def create_car(video:str, parking_zone:ParkingZone, action='enter') -> Car:
        
        # memoize the cars that get created
        car_index = f"{video}-{action}"
        if car_index in cars_record:
            return cars_record[car_index]
        
        # load the frames and record, and compute the feature for the car
        frames, record = dm.load_frames_and_record(video)
        record['in_driving_region'] = parking_zone.compute_in_driving_region(record)
        if len(record['track_id'].unique()) > 1:
            print(f"Error: More than one track_id in {video}")
            return None
        feature = parking_zone.create_feature(frames, record, action)
        car = Car()
        car.set_feature(feature)
        
        # memoize this car, so that it doesn't need to be recreated
        cars_record[car_index] = car
        
        return car
    
    # load a list of all of the videos that can be used for the zones
    test_set = load_test_set(test_set_label)
    enter_videos = get_enter_videos(test_set)
    leave_videos = get_leave_videos(test_set)
    pair_videos = get_video_pairs(test_set)
    
    results:list[dict] = []
    
    # load the ParkingZone
    parking_zone = dm.get_parking_zone_from_zoneID(zoneID)
    
    # run n_tests for each enter/leave pair
    pbar = tqdm(total=len(pair_videos), position=1)
    for pair_index, (enter_video, leave_video) in enumerate(pair_videos):
        
        curr_enter_videos:list[str] = (
            enter_videos + 
            [pv[0] for i, pv in enumerate(pair_videos) if i != pair_index]
        )
        
        # create all of the enter cars
        enter_cars = [create_car(video, parking_zone) for video in curr_enter_videos]
        enter_cars = [car for car in enter_cars if car is not None]
            
        # create the current enter_car and leave_car
        enter_car = create_car(enter_video, parking_zone)
        leave_car = create_car(leave_video, parking_zone, action='leave')
        enter_car = deepcopy(enter_car)
        leave_car = deepcopy(leave_car)
        enter_car.set_carID("target")
        leave_car.set_carID("target")
        
        # run a test for every possible selection of zone_size-1 cars
        tests = list(map(list, combinations(enter_cars, zone_size-1)))
        # if there are more than max_tests tests, then randomly select 
        # a 'max_tests' subset of tests
        if len(tests) > max_tests:
            tests = random.sample(tests, max_tests)
        n_tests = len(tests)
        
        counter = 0
        descr_str = f"{(counter:=counter+1)}/{n_tests} tests for pair#{pair_index}"
        pbar.set_description("{:^30}".format(descr_str))
        
        # run each of the tests
        for curr_test_cars in tests:
            parking_zone.clear_cars()
            
            random.shuffle(curr_test_cars)
            for curr_car in curr_test_cars:
                parking_zone.add_car(curr_car)
                
            # add the enter_car, shuffle the parking_zone cars, and get the match
            parking_zone.add_car(enter_car)
            parking_zone.shuffle_cars()
            matched_car = parking_zone.get_match(leave_car, k=k)
            
            # determine whether or not the parking zone was accurate and log the results
            is_match = matched_car == enter_car
            results.append({
                'pair_index': pair_index,
                'pair_enter_video': enter_video,
                'pair_leave_video': leave_video,
                'matched_car': repr(matched_car),
                'enter_car': repr(enter_car),
                'leave_car': repr(leave_car),
                'is_match': is_match,
            })
            
            descr_str = f"{(counter:=counter+1)}/{n_tests} tests for pair#{pair_index}"
            pbar.set_description("{:^30}".format(descr_str))
        pbar.update(1)
            
    pbar.close()
    
    return results
            
            
if __name__ == "__main__":
    
    model_label = "8hist_centerlastfirst_kandzerocenters"
    
    input(f"Is model_label='{model_label}' correct? Press Enter to continue, CTRL+C to quit ...")
    
    zoneID = "JFSBP1_center_east-region_1-south_east"
    parking_zone:ParkingZone = dm.get_parking_zone_from_zoneID(zoneID)
    
    zone_sizes = list(range(5, 13+1, 2))
    params = [
        (zone_size, k)
        for zone_size in zone_sizes
        for k in range(3, zone_size+1)
    ]
    
    pbar_main = tqdm(total=len(params), leave=True, position=0)

    for zone_size, k in params:
        pbar_main.set_description(f"zone_size: {zone_size}, k: {k}")
        
        results_path = f"tests/results/{model_label}/{zoneID}/parking_zone_inference_{zone_size}_{k}.csv"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if os.path.exists(results_path):
            continue
        
        results = test_parking_zone_inference(zone_size, zoneID, zoneID, k=k)
        results_df = pd.DataFrame(results)
        results_df['zone_size'] = zone_size
        results_df['k'] = k
        try:
            results_df.to_csv(results_path, index=False)
        except KeyboardInterrupt as ke:
            print("KeyboardInterrupt caught, don't CTRL+C again. Terminating...")
            results_df.to_csv(results_path, index=False)
            raise ke
        
        pbar_main.update(1)

    results = test_parking_zone_inference(7, zoneID, zoneID, k=5)