import os
import pandas as pd
from datetime import datetime

from config import ZONING_DATA_PATH

enter_leave_pairs = pd.read_csv(ZONING_DATA_PATH)

def sort_videos_chronologically(video_names):
    def extract_timestamp(video_name):
        timestamp_str = video_name[4:-10]
        return datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
    
    return sorted(video_names, key=extract_timestamp)

def get_timestamp_from_filename(filename):
    timestamp_str = filename.split(".")[0][4:23]
    try:
        return datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        return None

def get_vids_between(enter_vid_path, leave_vid_path):
    enter_day_folder = os.path.dirname(enter_vid_path) + "/"
    leave_day_folder = os.path.dirname(leave_vid_path) + "/"
    
    enter_vid = enter_vid_path.split("/")[-1]
    leave_vid = leave_vid_path.split("/")[-1]
    
    enter_timestamp = get_timestamp_from_filename(enter_vid)
    leave_timestamp = get_timestamp_from_filename(leave_vid)
    
    vids_between = []
    
    for filename in os.listdir(enter_day_folder):
        timestamp = get_timestamp_from_filename(filename)
        if timestamp is not None and enter_timestamp < timestamp < leave_timestamp and filename in enter_leave_pairs["video"].values:
            vids_between.append(filename)
    if leave_day_folder != enter_day_folder:
        for filename in os.listdir(leave_day_folder):
            timestamp = get_timestamp_from_filename(filename)
            if timestamp is not None and enter_timestamp < timestamp < leave_timestamp and filename in enter_leave_pairs["video"].values:
                vids_between.append(filename)
    return sort_videos_chronologically(vids_between)