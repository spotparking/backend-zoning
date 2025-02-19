import numpy as np 
import pandas as pd
from pathlib import Path
import cv2


def create_feature(frames:list[np.ndarray], record:pd.DataFrame) -> np.ndarray:\
    
    # 'take driving pics'
    
    # get center point
    
    # compute color histogram
    
    # width height at each driving picture
    
    return np.array([])

def average_color_histogram(folder_path):
    folder_path = Path(folder_path)
    hist_sum = None
    count = 0

    for image_path in folder_path.glob("*.jpg"):
        image = cv2.imread(str(image_path))

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
    
def compare_histograms(hist1, hist2):
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)