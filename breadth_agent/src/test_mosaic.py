from modules.geometry import GeometryComposition, GeometryProcessing, HomographyApplication
from modules.features import FeatureDetection, FeatureMatching

# Construct Modules with Initialized Arguments
image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\prompt_review\\image_data\\mosaic-data"

feature_detector = FeatureDetection(image_path)
feature_matcher = FeatureMatching("flann", type = 'partial', detector="sift")

geom_processing = GeometryProcessing()
geom_composition = GeometryComposition(image_path)
homography_app = HomographyApplication(image_path, "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent")

detected_features = feature_detector()

matched_features = feature_matcher(detected_features)

homographies = geom_processing("homography", "full", matched_features)

composed_homographies = geom_composition("homography", homographies)

# results = homography_app(composed_homographies)



