


def create_zone(license_plate, track_id, enter_vid_path):
    zone = pd.DataFrame(columns=["license_plate", "drive_pics_folder_path"])
    date_time = str(get_timestamp_from_filename(os.path.basename(enter_vid_path)))
    drive_pics_folder_path = f"first_zone/{date_time}/{license_plate}/"
    
    record_path = enter_vid_path.replace(".mp4", ".csv")
    record_path = record_path.replace("MP4s", "CSVs")
    record = pd.read_csv(record_path)
    rel_track_record = record[record["track_id"] == track_id]
    take_drive_pics(rel_track_record, enter_vid_path, drive_pics_folder_path)

    new_car = pd.DataFrame({'license_plate': license_plate, 'drive_pics_folder_path': drive_pics_folder_path}, index=[0])
    zone = pd.concat([zone, new_car], ignore_index=True)
    return zone

def add_car_to_zone(zone, license_plate, track_id, enter_vid_path):
    date_time = str(get_timestamp_from_filename(os.path.basename(enter_vid_path)))
    drive_pics_folder_path = f"first_zone/{date_time}/{license_plate}/"
    
    record_path = enter_vid_path.replace(".mp4", ".csv")
    record_path = record_path.replace("MP4s", "CSVs")
    record = pd.read_csv(record_path)
    rel_track_record = record[record["track_id"] == track_id]
    take_drive_pics(rel_track_record, enter_vid_path, drive_pics_folder_path)
    
    new_car = pd.DataFrame({'license_plate': license_plate, 'drive_pics_folder_path': drive_pics_folder_path}, index=[0])
    zone = pd.concat([zone, new_car], ignore_index=True)
    return zone

def remove_car_from_zone(zone, predicted_license_plate):
    folder_path = Path("./") / zone.loc[zone["license_plate"] == predicted_license_plate, "drive_pics_folder_path"].values[0]
    path_str = str(folder_path)
    path_str = path_str.rsplit("/", 1)[0] + "/"
    folder_path = Path(path_str)
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)
    zone = zone[zone["license_plate"] != predicted_license_plate]
    return zone