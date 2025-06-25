from modules.utilities.utilities import CalibrationReader
from modules.features import FeatureDetectionSIFT
from modules.featurematching import FeatureMatchFlannTracking
from modules.visualize import VisualizeScene
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"

# Feature Module Initialization
calibration_data = CalibrationReader(calibration_path).get_calibration()
feature_detector = FeatureDetectionSIFT(image_path=image_path)
feature_tracker = FeatureMatchFlannTracking(detector='sift')



# Solution Pipeline 
detected_features = feature_detector()

tracked_features = feature_tracker(detected_features)

print(tracked_features.data_matrix.shape)
print(tracked_features.point_count)

image_path = sorted(glob.glob(image_path + "\\*"))

for i in range(1000):
    if (tracked_features.access_point3D(i).shape[0] > 4):
        print(f"Point {i}:", tracked_features.access_point3D(i))
        print()
        fig,ax = plt.subplots(1, tracked_features.access_point3D(i).shape[0], figsize=(30, 15))

        for j in range(tracked_features.access_point3D(i).shape[0]):
            img = cv2.imread(image_path[j])
            ax[j].imshow(img)
            xx = tracked_features.access_point3D(i)[j, 1]
            yy = tracked_features.access_point3D(i)[j, 2]
            circ = Circle((xx,yy),20)
            ax[j].add_patch(circ)
        
        plt.tight_layout()
        plt.show()

        break