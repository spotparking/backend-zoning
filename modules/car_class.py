import numpy as np

class Car:
    def __init__(self):
        pass
        
    def set_features(self, features:np.ndarray):
        self.ave_hist = features[0]
        self.center_pt = features[1]