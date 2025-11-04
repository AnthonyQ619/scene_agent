from src.agent.prompt_agent import PromptEnhancmentAgent
from src.agent.coding_agent import CodeAgent
from src.agent.code_refinement_agent import CodeRefine


def main():
    path_for_testing = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\generated_code'
    instruction_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent_utils\\agent_instructions\\prompt_enh_examples.txt'
    full_length_context_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent_utils\\script_context\\full_context_scripts.txt'
    embed_vectors_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent_utils\\script_context\\embed_description_list.txt'
    api_path = 'C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\src\\agent_utils\\tool_context'
    first_agent_test = PromptEnhancmentAgent(instruction_path=instruction_path,
                                             model = 'gpt-5-2025-08-07')
    second_agent_test = CodeAgent(full_length_path=full_length_context_path, 
                                  embedding_database_path=embed_vectors_path,
                                  response_format='code',
                                  model='gpt-5-2025-08-07')
    # third_agent_task = CodeRefine(path_for_testing, api_path=api_path,
    #                               model='gpt-5-2025-08-07')


    # Final steps to include in prompt enhancment step
    # TODO: Set prompt enhancer llm to JSON with a split of answer into problem statement
    # and problem solution (Steps). With this in mind, reshape answer to this format
    # -> response_prompt[problem_statement] + data_description + response_prompt[problem_sol]
    # Design breaker statement for similarity check of full_length in-context examples to only
    # include problem statement and scene description.
    # TODO: Explore including scene description into prompt enhancment phase (step 2 of agent)

    temp_prompt = """
This is a video of an outdoor scene around a statue. I want to create a 3D representation
of this data, focused on the statue.
C:\\Users\\Anthony\\Documents\\Projects\\datasets\\Structure-from-Motion\\sfm_dataset
"""

    output, problem_statement = first_agent_test(temp_prompt)

    print(output)
    temp_file_propmt = open(path_for_testing + "\\generated_prompt.txt", 'w')

    temp_file_propmt.write(output)

    temp_file_propmt.close()

    code = second_agent_test(output, problem_statement)

    temp_file = open(path_for_testing + "\\temp_file.py", 'w')

    temp_file.write(code)

    temp_file.close()

    # script_file = 'temp_file.py'
    
    # third_agent_task(script_file_name=script_file)

if __name__ == "__main__":
    main()