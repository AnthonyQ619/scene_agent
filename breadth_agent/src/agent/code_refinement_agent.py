from openai import OpenAI
from sceneprogllm import LLM
import subprocess
import glob
import os 
import sys
import re

class Executor():
    def __init__(self, script_dir: str, output_file: str):
        if os.path.isdir(script_dir):
            self.dir = script_dir   # Path for temp directory
        else:
            Exception("No Directory or Python File exists")

        self.log_file = os.path.join(self.dir, output_file)


    def __call__(self, script_file: str, script_code: str):
        script_path = os.path.join(self.dir, script_file)
        if not os.path.isfile(script_path):
            Exception("No Directory or Python File exists")
        
        # Create Code
        temp_file = open(script_path, 'w')
        temp_file.write(script_code)
        temp_file.close()


        # Change current working directory
        os.chdir(self.dir)
        current_env = os.environ.copy()

        with open(self.log_file, "w") as f:
            print("🚀 Running script...")
            result = subprocess.run(
                [sys.executable, script_file],
                env=current_env,
                stdout=f,      # send standard output to file
                stderr=f       # send error output to file
            )

            f.close()

        log_file = open(self.log_file, 'r')

        output = log_file.read().strip()
        
        log_file.close()
        self.cleanup(self.log_file)

        if result.returncode == 0:
            return output, True
        else:
            return output, False
    
    def cleanup(self, log_file_path: str):
        if os.path.exists(log_file_path):
            os.remove(log_file_path)

class CodeRefine():
    def __init__(self, script_dir : str, 
                 model: str | None = None, 
                 refine_desc: str | None = None,
                 debug_desc: str | None = None,
                 use_cache_debug: bool = False,
                 temperature_debug: float = 0.8,
                 temperature_refine: float = 0.8,
                 api_path=None,
                 max_attempts = 5,
                 ):
        
        if model is None:
            self.model = "gpt-4o-mini"
        else:
            self.model = model

        self.MAX_ATTEMPTS = max_attempts
        self.exec = Executor(script_dir, "sfm_log.txt")
        if api_path is not None:
            api_files = sorted(glob.glob(api_path + "\\*"))
            self.api_desc = ""

            for file in api_files:
                context = open(file, 'r')
                examples = context.read()
                context.close()
                self.api_desc += examples + "\n"
        else:
            self.api_desc = None

        header = f"""
You should go through the code and find the errors including those caused by wrong use of the API. Then you must respond with the corrected code.
Only add the code that is necessary to fix the errors. Do NOT remove any original code in the script! Only change code with obvious errors for wrong use 
of API or incorrect python syntax!
"""
        default_desc = """
First identify the errors and then respond with the corrected code. You should also pay attention to the exceptions raised while running the code and find ways to fix them. 
You are not supposed to change placement values or settings in the code, but only watch out for reasons due to which the code may crash!
Also, you don't have to worry about importing additional modules. They are already imported for you.In case the code includes imports, make sure that debugged code includes the same imports.
Make sure that the debugged code does what the original code was intended to do and that it is not doing anything extra. The only thing you have to do is to fix the errors.
"""     
        if debug_desc is None:
            debug_header = f"""
You should go through the code as well as the traceback and find the errors including those caused by wrong use of the API. 
Then you must respond with the corrected code.
"""
            example = """

Example:
    code:
    a = 1+1
    b = 1+a
    c = 1++b
    print("Hello, world!"

    feedback:
    Fix the instantiation of variable "c" to include only one "+" operator.
    Fix the print statement to include a closing parantheses ")"

    debugged code:
    a = 1+1
    b = 1+a
    c = 1+b
    print("Hello, world!")
"""
            self.debug_desc = debug_header + default_desc + example
        else:
            self.debug_desc = debug_desc

        if refine_desc is None:
            example = """
            
Example:
    code:
    a = 1+1
    b = 1+a
    c = 1++b
    print("Hello, world!"

    debugged code:
    a = 1+1
    b = 1+a
    c = 1+b
    print("Hello, world!")
"""
            self.refine_desc = header + default_desc + example
        else:
            self.refine_desc = refine_desc

        system_desc = self.api_desc + self.refine_desc
        self.code_refine= LLM(name='refine_llm', 
                              system_desc=system_desc, 
                              response_format='code', 
                              use_cache=False,
                              model_name=self.model,
                              temperature = temperature_refine)

        system_desc = self.api_desc + self.debug_desc
        self.debugger = LLM(name='trace_debugger', 
                            system_desc=system_desc, 
                            response_format='code',
                            use_cache=use_cache_debug,
                            model_name=self.model,
                            temperature = temperature_debug)

        feedback_gen_string = f"""
        You are supposed to go through the stdout and respond whether there are 
        any errors or not. If there are errors, generate feedback necessary to correct 
        code using either API information for errors or standard python coding to fix
        the errros that exist in the standard output. DO NOT EDIT THE CODE. Generate text
        feedback of what SHOULD be done to fix the code. If you see errors, respond in a 
        JSON format with 'errors': 'True' and 'feedback': 'your response'. In case you 
        don't see any errors (ignore warnings!) respond in a JSON format with 
        'errors': 'False' and 'feedback': 'Not Applicable'.
        """

        feedback_desc = feedback_gen_string + self.api_desc
        self.feedback_gen = LLM(name='feedback_llm', 
                                system_desc=feedback_desc, 
                                response_format='json',
                                json_keys=["errors:bool", "feedback:str"],
                                use_cache=False,
                                model_name=self.model)

        # Script Directory
        self.script_dir = script_dir

    def _write_new_script(self, script_path: str, script_code: str):
        temp_file = open(script_path, 'w')

        temp_file.write(script_code)

        temp_file.close()

    def __call__(self, script_file_name: str):
        
        # Initial Run
        script_path = os.path.join(self.script_dir, script_file_name)

        # Read Code from Script
        f_script = open(script_path, 'r')
        script = f_script.read()
        f_script.close()
        
        # Initial Code Refinement for any hallucination or coding errors.
        refined_code = self.code_refine(script)

        print(refined_code)
        # Initial Code Running Stage
        output_trace, result = self.exec(script_file=script_file_name, script_code=refined_code)

        print("OUTPUT")
        print(output_trace)

        print(result)
        print("END OUTPUT")

        # prompt = f"Input: \n {script}.\nErrors: {output_trace}."

        # feedback = self.feedback_gen(prompt)
        # print(feedback)
        for i in range(self.MAX_ATTEMPTS): # Maximum number of attempts
            prompt = f"Input: \n {refined_code}.\nErrors: {output_trace}."

            feedback = self.feedback_gen(prompt)
            print(f"Run {i} was executed with result of errors being {feedback['errors']}")
            if result and not feedback['errors']:
                print("SUCCESS! A full run was executed to completion!")
                break
            
            debug_prompt = f"Previous Script: {refined_code}.\nFeedback: {feedback['feedback']}"
            refined_code = self.debugger(debug_prompt)

            output_trace, result = self.exec(script_file=script_file_name, script_code=refined_code)

        return refined_code
        