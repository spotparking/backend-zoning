import os

import numpy as np
import cv2


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
        if (counter:=counter+1) % frame_skip != 0:
            continue
        frames.append(frame)
    # Release the video capture object
    cap.release()
    return np.array(frames)

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
    
    
    
        
        
    