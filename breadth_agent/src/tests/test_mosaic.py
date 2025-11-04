from modules.geometry import GeometryComposition, GeometryProcessing, HomographyApplication
from modules.optimization import OutlierRejection
from modules.features import FeatureDetection, FeatureMatching

import cv2

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\prompt_review\\image_data\\mosaic-data"

# Initialize Feature Detection Modules
feature_detector = FeatureDetection(image_path, type="orb")
feature_matcher = FeatureMatching("bf", type = 'partial', detector="orb")
outlier_rejector = OutlierRejection()

# Initialize Homography Estimation and Application Modules
geom_processing = GeometryProcessing()
geom_composition = GeometryComposition(image_path)
homography_app = HomographyApplication(image_path, "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent")

# Feature Detection Step
detected_features = feature_detector() # list[Points2D] each index correspond to an image scene
matched_features = feature_matcher(detected_features)
inlier_features = outlier_rejector(option = "fundamental", format="full", pts=matched_features)
# print (inlier_features[:10]) # list of [Points2D, Points2D] -> Points2D: keypoints, descriptors -> keypoints.x and keypoints.y 
# print(len(inlier_features))  # all image pairs
# print(len(inlier_features[0])) # size 2 which stores the above information 

# Estimate The Homography Matrix Across Image Pairs
homographies = geom_processing("homography", "full", inlier_features)


# H1,2, H2,3 .. Hn-1,n -> H1,2, H1,3, H
composed_homographies = geom_composition("homography", homographies)

# Create Image Based on Estimate Homographies
results = homography_app(composed_homographies)

cv2.imwrite("MOSAIC_EXAMPLE.jpg", results)


