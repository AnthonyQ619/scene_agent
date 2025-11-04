import glob
from tqdm import tqdm
import os
import sys
from openai import OpenAI
from modules.utilities.utilities import CalibrationReader
from modules.features import (FeatureDetectionSIFT, 
                              FeatureDetectionSP, 
                              FeatureDetectionORB, 
                              FeatureDetectionFAST)
from modules.featurematching import (FeatureMatchFlannTracking, 
                                     FeatureMatchLoftrPair,
                                     FeatureMatchLightGlueTracking, 
                                     FeatureMatchSuperGlueTracking, 
                                     FeatureMatchLightGluePair, 
                                     FeatureMatchSuperGluePair,
                                     FeatureMatchRoMAPair)
from modules.camerapose import (CamPoseEstimatorEssentialToPnP, 
                                CamPoseEstimatorVGGTModel)
from modules.scenereconstruction import Sparse3DReconstructionMono
from modules.visualize import VisualizeScene
import numpy as np

CURRENT_PATH = str(os.path.dirname(__file__))

def build_context_string(module_tool):
    name = module_tool.module_name
    description = module_tool.description
    example = module_tool.example


    context_string = f"""
Tool Name: {name}

Tool Description: {description}

Tool Example: {example}

"""
    
    return context_string

def build_tool_context_file(filename: str, set_of_tools: list, tool_type:str):
    special_breaker = "===+&+===\n"
    if not os.path.isdir(CURRENT_PATH + "\\tool_context"):
        os.mkdir(CURRENT_PATH + "\\tool_context")
    file_to_write = open(CURRENT_PATH + "\\tool_context\\" + filename, 'w')

    for i in tqdm(range(len(set_of_tools))):
        tool = set_of_tools[i]
        if i == 0:
            starting_string = f"This is a file containing the {tool_type} modules.\n"
            context = build_context_string(tool)

            full_string = starting_string + special_breaker + context + special_breaker
        else:
            context = build_context_string(tool)
            full_string = context + special_breaker

        file_to_write.write(full_string)
    
    file_to_write.close()

def build_full_length_context_file(filename: str, context_file_path: str):
    special_breaker = "\n===$&$===\n"
    if not os.path.isdir(CURRENT_PATH + "\\script_context"):
        os.mkdir(CURRENT_PATH + "\\script_context")

    file_to_write = open(CURRENT_PATH + "\\script_context\\" + filename, 'w')

    context_files = sorted(glob.glob(context_file_path + "\\test_sfm_t*"))
    print(context_files)
    for i in range(len(context_files)):

        file = context_files[i]
        script = open(file, 'r')

        content = script.read()
        
        if i == 0: 
            full_string = special_breaker + content + special_breaker
        else:
            full_string = content + special_breaker
        file_to_write.write(full_string)

        script.close()

    file_to_write.close()

def build_embedded_description_db(file_path: str):
    assert os.path.exists(file_path)
    client = OpenAI()
    if not os.path.isdir(CURRENT_PATH + "\\script_context"):
        os.mkdir(CURRENT_PATH + "\\script_context")

    file_to_read = open(file_path, 'r')
    embed_file = open(CURRENT_PATH + "\\script_context\\embed_description_list.txt", 'w')

    content = file_to_read.read()

    split_db = content.split("===$&$===")

    for i in tqdm(range(1, len(split_db) - 1)):
        desc = split_db[i].split("# ==#$#==")[0].lstrip().rstrip().replace('"""','').replace("\n", "")

        # if i == 1:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=desc,
            dimensions=256,
            encoding_format="float"
            )
        embed_file.writelines([str(response.data[0].embedding) + "\n", desc + "\n"])

    embed_file.close()
    file_to_read.close()

def tool_building():
    image_path = "C:\\Users\\Anthony\\Documents\\Projects\datasets\\Structure-from-Motion\\sfm_dataset"
    calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\calibration.npz"
    calibration_data = CalibrationReader(calibration_path).get_calibration()

    features = [FeatureDetectionSIFT(image_path=image_path), 
                FeatureDetectionORB(image_path=image_path),
                FeatureDetectionSP(image_path=image_path)]
    
    matchers = [FeatureMatchFlannTracking(), 
                FeatureMatchLoftrPair(image_path),
                FeatureMatchLightGlueTracking(), 
                FeatureMatchSuperGlueTracking(), 
                FeatureMatchLightGluePair(), 
                FeatureMatchSuperGluePair(),
                FeatureMatchRoMAPair(image_path)]
    
    camera_pose_est = [CamPoseEstimatorEssentialToPnP(calibration_data, image_path=image_path),
                       CamPoseEstimatorVGGTModel(image_path=image_path, calibration=calibration_data)]
    
    scene_estimators = [Sparse3DReconstructionMono(calibration=calibration_data, image_path=image_path)]

    build_tool_context_file('feature_context.txt', features, "Features")
    build_tool_context_file('feature_matching_context.txt', matchers, "Feature Matcher")
    build_tool_context_file('camera_pose_context.txt', camera_pose_est, "Pose Estimation")
    build_tool_context_file('scene_const_context.txt', scene_estimators, "Scene Reconstruction")


if __name__ == "__main__":

    arg = sys.argv[1]
    if arg == "tool":
        tool_building()
    elif arg == "script":
        path_to_files = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\tests\\context_tests"
        build_full_length_context_file("full_context_scripts.txt", path_to_files)
    elif arg == "embed" or arg =="embedding":
        path_to_file = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent_utils\\script_context\\full_context_scripts.txt"
        build_embedded_description_db(path_to_file)
    elif arg == "test" or arg == "t":
        path_to_file = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent_utils\\script_context\\embed_description_list.txt"

        file_to_read = open(path_to_file, 'r')
        file_lines = file_to_read.readlines()
        # print(len(file_lines))
        for i in range(0, len(file_lines), 2):
            embed_vec = file_lines[i]
            embed_vec = list(map(float, embed_vec.replace('[', '').replace(']', '').split(',')))
            desc = file_lines[i+1]
            print(np.array(embed_vec).dtype)
