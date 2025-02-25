import numpy as np

class Car:
    def __init__(self):
        pass
        
    def set_feature(self, feature:np.ndarray):
        """ this sets the Car's feature vector, which acts as an abstract representation of the car (visually,
        positionally, etc.). The feature is an (n,) numpy array where n is the number of numeric features. 
        Currently feature is the concatenation of:
          - average hue histogram (b*,)
          - average saturation histogram (b*,)
          - average value histogram (b*,)
          - cx (float)
          - cy (float)
          *b is the number of bins in the histogram, currently 256
          
        the feature comes from ParkingZone.create_feature
        """
        if not isinstance(feature, np.ndarray):
            raise TypeError("feature must be a numpy array")
        if not len(feature.shape) == 1:
            raise ValueError("feature must be a 1D array")
        self.feature = feature
        
    def get_center_pt(self):
        return tuple(self.feature[-2:])
        