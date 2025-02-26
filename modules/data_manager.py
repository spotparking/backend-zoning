import os
from typing import Tuple, Dict

import dotenv
import yaml

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
    
