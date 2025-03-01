import os
from typing import Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import dotenv
import yaml
from modules.parking_zone_class import ParkingZone

dotenv.load_dotenv()

# download the dataset from 
# https://drive.google.com/drive/folders/1gApCFWWvYCphGbjIl55OWEzvLnWY94YX?usp=sharing
# and unzip the jfsb_selection.zip fold somewhere on your computer
# then create a '.env' file in the root of this repo, and add this line
# DATA_PATH="<path to jfsb_selection>"
# for example:
# DATA_PATH="/home/development/DATA"
DATA_PATH = os.getenv("DATA_PATH")

def datapath(filename:str) -> str:
    path = os.path.join(DATA_PATH, filename)
    if os.path.exists(path):
        return path
    else:
        raise ValueError(f"datapath('{filename}') -> '{path}' does not exist")

############################################################
#                 JFSB SELECTION DATA PATHS                #
############################################################

def get_jfsb_tracking_csv_path(video_name:str) -> str:
    return datapath(f"jfsb_selection/tracking_csvs/{video_name}.csv")

def get_jfsb_motion_csv_path(video_name:str) -> str:
    return datapath(f"jfsb_selection/motion_csvs/{video_name}.csv")

def get_jfsb_video_path(video_name:str) -> str:
    return datapath(f"jfsb_selection/videos/{video_name}.mp4")

def get_jfsb_info_csv_path() -> str:
    return datapath(f"jfsb_selection/info.csv")

def get_date_str_cam_label(video_name:str) -> Tuple[str, str]:
    splits = video_name.split("_")
    date_str = splits[0]
    cam_label = "_".join(splits[1:])
    return date_str, cam_label

############################################################
#                         SETTINGS                         #
############################################################

def get_settings_path(cam_label:str) -> str:
    settings_path = f"settings/{cam_label}.yaml"
    if os.path.exists(settings_path):
        return settings_path
    else:
        raise ValueError(f"get_settings_path('{cam_label}') -> '{settings_path}' does not exist")

def get_settings(cam_label:str) -> dict:
    yaml_settings_path = get_settings_path(cam_label)
    with open(yaml_settings_path, "r") as f:
        settings = yaml.safe_load(f)
    return settings

def parse_zoneID(zoneID:str, full_cam_label=True) -> Tuple[str, str, str]:
    """ takes in a zoneID like 'JFSBP1_center_east-region_0-north_east' and returns
    a tuple of the cam_label, region_id, and zone_name 
    """
    splits = zoneID.split("-")
    cam_label = splits[0]
    if full_cam_label and "JFSB" in cam_label and "000004" not in cam_label:
        cam_label = f"000004-1-{cam_label}"
    region_id = splits[1]
    zone_name = splits[2]
    return cam_label, region_id, zone_name

def get_zoneID_from_parts(cam_label:str, region_id:str, zone_name:str) -> str:
    """ takes in the cam_label, region_id, and zone_name and returns the zoneID
    like 'JFSBP1_center_east-region_0-north_east' 
    """
    if "JFSB" in cam_label:
        cam_label = cam_label.replace("000004-1-", "")
    return f"{cam_label}-{region_id}-{zone_name}"

def get_zone_settings(zoneID:str, settings:dict=None) -> dict:
    """ takes in a zoneID like 'JFSBP1_center_east-region_0-north_east', loads
    the settings/<cam_label>.yaml file, and returns the settings for just the specified
    zone 
    """
    # parse the zoneID
    cam_label, region_id, zone_name = parse_zoneID(zoneID)
    
    # the user can specify the settings file for quicker loading, or
    # the settings can be auto-loaded in this function
    if settings is None:
        settings = get_settings(cam_label)
    
    zones:dict = settings.get('zones', {})
    # the settings zones indexer doesn't have the '000004-1-' prefix and doesn't have the 
    # zone_name at the end of them
    indexer = "-".join(zoneID.split("-")[:-1])
    zone:dict = zones.get(indexer, {})
    
    if zone.get('zone_name', '') == zone_name:
        return zone
    else:
        raise ValueError(
            f"get_zone_settings('{zoneID}') -> {zone} which doesn't "
            f"have zone_name={zone_name}")
        
def get_camera_resolution(cam_label:str, settings:dict=None) -> Dict[str, int]:
    """ takes in a cam_label like '000004-1-JFSBP1_center_east' and returns a 
    dictionary in the format:
    {
        "height": 1944,
        "width": 2592,
    }"""
    if settings is None:
        settings = get_settings(cam_label)
    return settings['resolution']

############################################################
#                 FRAMES AND RECORD LOADING                #
############################################################

def extract_frames(video_path, frame_skip=None) -> list[np.ndarray]:
    """
    Extracts all frames from a video and returns the frames as np.ndarrays
    inside a list.

    Parameters:
        video_path (str): Path to the input video file.
        
    frame_skip==None (default) -> every frame
    frame_skip==1 -> every frame
    frame_skip==2 -> every other frame
    frame_skip==3 -> every third frame
    ...
    """
    if frame_skip is None:
        frame_skip = 1
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return []
    # Read all frames
    counter = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (counter:=counter+1) % frame_skip == 0:
            frames.append(frame)
    # Release the video capture object
    cap.release()
    return frames

def _load_frames_and_record_handle_edgecases(video_name:str, record:pd.DataFrame) -> pd.DataFrame:
    if video_name == "2025-02-16-09-05-29_000004-1-JFSBP1_center_west":
        record = record[record['track_id'] == 2]
    # elif video_name == "2025-02-15-15-19-11_000004-1-JFSBP1_center_west":
    #     record.loc[record.index.values[0], ['br_x', 'cx', 'cx_ma', 'w']] = record.loc[record.index.values[0], ['tl_x', 'br_x', 'cx', 'cx_ma']] + 100
    #     record.loc[record.index.values[1], ['br_x', 'cx', 'cx_ma', 'w']] = record.loc[record.index.values[1], ['tl_x', 'br_x', 'cx', 'cx_ma']] + 50
    # elif video_name == "2025-02-15-14-27-43_000004-1-JFSBP1_center_west":
    #     record.loc[record.index.values[-1], ['br_x', 'cx', 'cx_ma', 'w']] = record.loc[record.index.values[0], ['tl_x', 'br_x', 'cx', 'cx_ma']] + 200
    #     record.loc[record.index.values[-2], ['br_x', 'cx', 'cx_ma', 'w']] = record.loc[record.index.values[1], ['tl_x', 'br_x', 'cx', 'cx_ma']] + 100
    return record

def load_frames_and_record(video_name:str) -> Tuple[list[np.ndarray], pd.DataFrame]:
    """ loads the frames and record for a video_name like 
    '2025-02-14-18-43-14_000004-1-JFSBP1_center_west' 
    """
    tracking_csv_path = get_jfsb_tracking_csv_path(video_name)
    motion_csv_path = get_jfsb_motion_csv_path(video_name)
    video_path = get_jfsb_video_path(video_name)
    
    # load the frames from the video
    if os.path.exists(video_path):
        frames = extract_frames(video_path)
        
        # load the motion_csv or tracking_csv (priority to the motion_csv)
        if os.path.exists(motion_csv_path):
            record = pd.read_csv(motion_csv_path)
        elif os.path.exists(tracking_csv_path):
            record = pd.read_csv(tracking_csv_path)
        else:
            raise ValueError(
                f"load_frames_and_record('{video_name}') -> "
                f"neither motion_csv_path='{motion_csv_path}' nor "
                f"tracking_csv_path='{tracking_csv_path}' exist"
            )
    else:
        raise ValueError(
            f"load_frames_and_record('{video_name}') -> "
            f"video_path='{video_path}' does not exist"
        )
        
    # handle edge cases...
    record = _load_frames_and_record_handle_edgecases(video_name, record)
    if record is None:
        return [], pd.DataFrame()
        
    # only keep the tracks that drove into a parking region
    # sadly the data doesn't specify which one...
    # record = record[record['any_in_parking_region']==True]
    return frames, record

def get_parking_zone_from_zoneID(zoneID:str) -> ParkingZone:
    # load the zone_settings based on the zoneID
    cam_label, region_id, zone_name = parse_zoneID(zoneID)
    settings = get_settings(cam_label)
    zone_settings = get_zone_settings(zoneID, settings=settings)
    
    # load the coordinates as pixels instead of floats
    resolution = get_camera_resolution(cam_label, settings=settings)
    height = resolution['height']
    width = resolution['width']
    coordinates = ParkingZone.points_float_to_pix(zone_settings['points'], height, width)
    
    # create and return the ParkingZone object
    return ParkingZone(zoneID, cam_label, coordinates, [])
        
    
    
