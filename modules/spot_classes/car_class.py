import numpy as np
import uuid
import ast
import re

class Car:
    def __init__(self, carID:str=None):
        self.set_carID(carID)
        self.feature:np.ndarray = None
        
    def set_carID(self, carID:str=None):
        if carID is None:
            carID = str(uuid.uuid4())
        self.carID = carID
        
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
        self.feature:np.ndarray = feature
        
    def get_feature(self):
        if self.feature is None:
            raise ValueError("feature has not been set for this car")
        return self.feature  
        
    def get_ave_hist(self):
        return self.feature[:-2]
        
    def get_center_pt(self):
        return tuple(self.feature[-2:])
    
    def __eq__(self, other:'Car'):
        if not isinstance(other, Car):
            return False
        return self.carID == other.carID
    
    def __repr__(self):
        return f"Car(carID={repr(self.carID)},feature={repr(self.feature.tolist())})"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def from_repr(repr_str: str) -> 'Car':
        """ if you have a car object, then you can use this function to do:
        car = Car()
        car.set_feature(feature)
        car_repr_str = repr(car)
        ...
        car = Car.from_repr(car_repr_str)
        """
        # pars the repr_str
        match = re.search(r"Car\(carID='(.*?)',feature=(.*?)\)", repr_str)
        if not match:
            raise ValueError(f"Invalid repr string '{repr_str}'")
        # create a car with the same carID
        carID = match.group(1)
        car = Car(carID=carID)
        # extract the feature array from the repr string and set it
        feature_list_repr = match.group(2)
        feature = ast.literal_eval(feature_list_repr)
        feature = np.array(feature)
        car.set_feature(feature)
        return car
        