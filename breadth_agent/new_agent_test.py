from src.agent.prompt_agent import PromptEnhancmentAgent
from src.agent.coding_agent import CodeAgent
from src.agent.programmer_agent import ProgramManagerLLM, ProgrammerAgent
from src.agent.code_refinement_agent import CodeRefinementLLM
from src.agent.core.imageanalyzer import ImageAnalyzer
from src.agent.core.generator import Generator
# from src.agent.code_refinement_agent import CodeRefine
import time


def main():
    script_dir_testing = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\generated_code'
    instruction_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\agent_details\\agent_instructions\\prompt_enh_examples.txt'
    full_length_context_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\agent_details\\script_context\\full_context_scripts.txt'
    embed_vectors_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\agent_details\\script_context\\embed_description_list.txt'
    api_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent\\agent_details\\tool_context'
    first_agent_test = PromptEnhancmentAgent(full_length_path=full_length_context_path,
                                             embedding_database_path=embed_vectors_path,
                                             instruction_path=instruction_path,
                                             model = 'gpt-5-2025-08-07')
    # second_agent_test = ProgramManagerLLM(api_directory=api_path)
    second_agent = ProgrammerAgent(model = "gpt-5-2025-08-07")
    third_agent = CodeRefinementLLM(script_dir=script_dir_testing, model = "gpt-5-2025-08-07", api_path=api_path)

    analyzer = ImageAnalyzer(model = "gpt-5-2025-08-07",
                             k_images=5)
    compiler = Generator(model="gpt-5-2025-08-07",
                         api_directory=api_path)
    # second_agent_test = CodeAgent(full_length_path=full_length_context_path, 
    #                               embedding_database_path=embed_vectors_path,
    #                               response_format='code',
    #                               model='gpt-5-2025-08-07')
    # third_agent_task = CodeRefine(path_for_testing, api_path=api_path,
    #                               model='gpt-5-2025-08-07')


    # Final steps to include in prompt enhancment step
    # TODO: Set prompt enhancer llm to JSON with a split of answer into problem statement
    # and problem solution (Steps). With this in mind, reshape answer to this format
    # -> response_prompt[problem_statement] + data_description + response_prompt[problem_sol]
    # Design breaker statement for similarity check of full_length in-context examples to only
    # include problem statement and scene description.
    # TODO: Explore including scene description into prompt enhancment phase (step 2 of agent)

#     temp_prompt = """
# Images: C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan30_normal_lighting
# """
    # image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\Family"
    # calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\calibration_new_2048.npz"
    # image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\Francis\\Francis"
    # calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\Tanks_and_Temples\\calibration_new_1920.npz"
    image_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\scan118_normal_lighting"
    calibration_path = "C:\\Users\\Anthony\\Documents\\Projects\\datasets\\sfm_dataset\\DTU\\calibration_DTU_new.npz"
    reconstruction_type = "Sparse Reconstruction"
    gpu_mem = "12gb"
    temp_prompt = {'images':image_path, 
                   'calibration':calibration_path, 
                   'recon_type':reconstruction_type, 
                   'gpu_mem':gpu_mem}

    print(temp_prompt)
    t0 = time.time()
    # scene_description, problem_statement = first_agent_test(temp_prompt)
    # result = analyzer(temp_prompt['images'])
    t1 = time.time()

    print("TIME:", t1-t0)
    # print(result)
    """
    log_file = open(script_dir_testing + "\\key_facts_during_gen.txt", "w")
    # context = open("C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\tests\\context_tests\\test_sfm_t1.py", 'r')
    # examples = context.read()
    # context.close()
    log_file.write("SCENE DESCRIPTION:\n" + scene_description + "\n")
    log_file.write("PROBLEM STATEMENT:\n" + problem_statement + "\n")
    log_file.write("TIMING OF FIRST AGENT BLOCK (PROBLEM/IMAGE ANALYSIS):\n" + str(t1-t0) + "\n")
    print(scene_description)
    print(problem_statement)
    
    print()
    print("Compiling Code...")
    t0 = time.time()
    code = second_agent(problem_statement)
    t1 = time.time()
    log_file.write("TIMING OF SECOND AGENT BLOCK (CODE GENERATION):\n" + str(t1-t0) + "\n")
    print("Verifying Code...")
    t0 = time.time()
    refined_code = third_agent(script_file_name="gen_code.py")
    t1 = time.time()

    log_file.write("TIMING OF THIRD AGENT BLOCK (CODE DEBUGGING/REFINEMENT):\n" + str(t1-t0) + "\n")
    # output = second_agent_test(scene_description + problem_statement + "\n" + examples)
    # print()
    # print("STEP 1", output.feature_step)
    # print()
    # print("STEP 2", output.matcher_step)
    # print()
    # print("STEP 3", output.pose_step)
    # print()
    # print("STEP 4", output.tracking_step)
    # print()
    # print("STEP 5", output.scene_est_step)
    # print()
    # print("STEP 6", output.optim_step)
    # temp_file_propmt = open(path_for_testing + "\\generated_prompt.txt", 'w')

    # temp_file_propmt.write(output)

    # temp_file_propmt.close()

    # code = second_agent_test(output, problem_statement)

    temp_file = open(script_dir_testing + "\\refined_gen_code.py", 'w')

    temp_file.write(refined_code)

    temp_file.close()

    # script_file = 'temp_file.py'
    
    # third_agent_task(script_file_name=script_file)"""

if __name__ == "__main__":
    main()