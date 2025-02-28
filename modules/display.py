import os

from IPython.display import clear_output
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

SPOT_YELLOW_BGR = (116, 227, 255) # spot YELLOW in BGR


def save_image(image, save_path):
    cv2.imwrite(save_path, image)

def show_image(image, window_name="Image", height=480, width=640, init_window=True, save_path=None, still_show=False):
    if isinstance(image, str):
        image = cv2.imread(image)
        
    if save_path is not None and isinstance(save_path, str) and not still_show:
        cv2.imwrite(save_path, image)
        return
    
    if init_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.moveWindow(window_name, 0, 0)
    
    cv2.imshow(window_name, image)
    
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        # wait for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
        
def get_image(image):
    """
    Loads an image from a file path or returns the image if it's already a numpy array.

    Parameters:
    image (numpy.ndarray or str): 
        - If a numpy array of shape (h, w, 3), it is returned as is.
        - If a string, it is treated as a file path to load the image.

    Returns:
    numpy.ndarray: The loaded image in BGR format.

    Raises:
    ValueError: If the image path does not exist.
    """
    if isinstance(image, str):
        if os.path.exists(image):
            image = cv2.imread(image)
        else:
            raise ValueError(f"Error: Could not find image at '{image}'")
    return image


def extract_first_frame(video_path) -> np.ndarray:
    """
    Extracts the first frame from a video and returns it as an np.ndarray.

    Parameters:
        video_path (str): Path to the input video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    # Read the first frame
    ret, frame = cap.read()
    # Release the video capture object
    cap.release()
    return frame

def extract_last_frame(video_path) -> np.ndarray:
    """
    Extracts the last frame from a video and returns it as an np.ndarray.

    Parameters:
        video_path (str): Path to the input video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    # Move to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    # Read the last frame
    ret, frame = cap.read()
    # Release the video capture object
    cap.release()
    return frame

def extract_frames(video_path, frame_skip=None) -> np.ndarray:
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
        return
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
    return np.array(frames)

def show_video(frames:list[np.ndarray], record:pd.DataFrame, track_id:int|str=None, track_id_col:str='track_id', plt_show=True):
    
    # check the track_id_col to make sure it is in the record
    if track_id_col not in record.columns:
        raise ValueError(f"track_id_col='{track_id_col}' not in record.columns")
    
    # filter the record to only be of the track_id
    if track_id is not None:
        record = record.loc[record[track_id_col] == track_id, :]
        if len(record) == 0:
            raise ValueError(f"track_id={track_id} not found in record")
    
    title_string = "video"
    
    # load the bounding boxes for the relevant tracks and map them to each frame
    frame_tracks = {}
    record['br_x'] = record['tl_x'] + record['w']
    record['br_y'] = record['tl_y'] + record['h']
    for iteration, frame in record.groupby('iteration'):
        frame = frame.loc[frame['confirmed'], :]
        bbox_df = frame[[track_id_col, 'confidence', 'tl_x', 'tl_y', 'br_x', 'br_y']]
        bbox_arr = bbox_df.to_numpy().astype(float)
        frame_tracks[int(iteration)-1] = bbox_arr
        
    # loop through the frames and draw the bounding boxes on each frame
    for i, frame in enumerate(frames):
        disp_frame = frame.copy()
        bbox = frame_tracks.get(i, None)
        if bbox is not None:
            for track_id, confidence, tl_x, tl_y, br_x, br_y in bbox:
                cv2.rectangle(disp_frame, (int(tl_x), int(tl_y)), (int(br_x), int(br_y)), SPOT_YELLOW_BGR, 5)
                cv2.putText(disp_frame, f"{track_id}", (int(tl_x), int(tl_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # show the video via matplotlib or cv2
        if plt_show:
            plt.title(title_string) if title_string is not None else None
            disp_frame = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
            plt.imshow(disp_frame)
            plt.axis('off')
            plt.gcf().set_size_inches(15, 8)
            plt.show()
            clear_output(wait=True)  # Clear the previous frame
        else:
            cv2.imshow(title_string, disp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(title_string, cv2.WND_PROP_VISIBLE) < 1:
                break

# def extract_frames(video_path: str):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     if not cap.isOpened():
#         raise ValueError(f"Could not open video file: {video_path}")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frames.append(frame)

#     cap.release()

#     return np.array(frames)
    
    
    
        
        
    