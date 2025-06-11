import cv2
import numpy as np
import sys
import re
sys.path.insert(0, "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\modules")
from DataTypes.datatype import Calibration


# function to convert calibration data into a readable numpy file

def convertToNPZ(path_name:str, file_name: str | None = None, np_data: list[np.ndarray] | None = None):
    ordered_keys = ['K1', 'Dist', 'K2', 'Dist2', 'Extrinsics']
    calibration = {}
    if file_name is not None:
        # File should store data line by line
        # K1: k11, k12, k13, k21, k22, ...
        # Dist: d1, d2, d3, d4, d5
        # K2: etc..
        # Dist2: d1, d2, ..
        # Extrinsics: R11, R12, ... , T1, T2, T3
        data = open(file_name)
        lines = data.readlines()

        for i in range(len(lines)):
            line = lines[i]
            
            cal_data = line.strip().split(',')
            cal_data[0] = cal_data[0].split(":")[1].lstrip()
            cal_data = np.array(cal_data, dtype=np.float32).reshape((3,3))
            calibration[ordered_keys[i]] = cal_data
            
            print(cal_data)

        print(calibration)
        calibrationer = Calibration(**calibration)

        print(calibrationer.K1)
        np.savez(path_name + "\\calibration.npz", **calibration)
    elif np_data is not None:
        for i in range(len(np_data)):
            calibration[ordered_keys[i]] = np_data[i]

        np.savez(path_name + "\\calibration.npz", **calibration)


class CalibrationReader:
    def __init__(self, file_path: str):
        #self.calibration = Calibration() # Think about given a subset to a full set of calibration data, how we can fill the calibration data like so by passing in certain arguments
        self.calibration = self.read_npz(file_path)

    def read_npz(self, file_path) -> Calibration:
        data = np.load(file_path)
        full_data = dict(data)
        calibration = Calibration(**full_data)

        return calibration
    
    def get_calibration(self) -> Calibration:
        return self.calibration
    

if __name__=='__main__':
    # text_path = 'C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.txt'
    path_name = 'C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz'
    # convertToNPZ(path_name=path_name, file_name=text_path)

    calibration = CalibrationReader(path_name).get_calibration()

    print(calibration.K1)






