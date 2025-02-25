import os

import cv2
import numpy as np


###################
# FEATURE HELPERS #
###################
def is_in_zone(x:int, y:int, zone:np.ndarray) -> bool:
    return cv2.pointPolygonTest(zone, (int(x), int(y)), measureDist=False) >= 0

def is_in_zone_vec(centers:np.ndarray, zone:np.ndarray) -> np.ndarray:
    return np.array([
        is_in_zone(x, y, zone)
        for x, y in centers
    ]).astype(bool)