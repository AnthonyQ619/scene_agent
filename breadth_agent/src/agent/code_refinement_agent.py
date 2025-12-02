from openai import OpenAI
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

import subprocess
import glob
import os 
import sys
import re

class Executor():
    def __init__(self, 
                 script_dir: str, 
                 output_file: str):
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

class CodeRefinementLLM():
    def __init__(self, script_dir : str, 
                 model: str | None = None, 
                 refine_desc: str | None = None,
                 debug_desc: str | None = None,
                 temperature_debug: float = 0.8,
                 temperature_refine: float = 0.8,
                 api_path=None,
                 max_attempts = 5,
                 ):
        
        if model is None:
            self.model_name = "gpt-5-mini"
        else:
            self.model_name = model

        api_key=os.getenv("OPENAI_API_KEY")

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
        # self.code_refine= LLM(name='refine_llm', 
        #                       system_desc=system_desc, 
        #                       response_format='code', 
        #                       use_cache=False,
        #                       model_name=self.model,
        #                       temperature = temperature_refine)
        self.sys_msg_refine = SystemMessage(self.api_desc + self.refine_desc + "Output executable Python Code only, no text output otherwise")

        self.model_refine = init_chat_model(model=self.model_name,
                                            api_key=api_key,
                                            temperature=temperature_refine)

        self.sys_msg_debug = SystemMessage(self.api_desc + self.debug_desc + "Output executable Python Code only, no text output otherwise")
        # self.debugger = LLM(name='trace_debugger', 
        #                     system_desc=system_desc, 
        #                     response_format='code',
        #                     use_cache=use_cache_debug,
        #                     model_name=self.model,
        #                     temperature = temperature_debug)
        self.model_debugger = init_chat_model(model=self.model_name,
                                              api_key=api_key,
                                              temperature=temperature_debug)


        class Feedback(BaseModel):
            errors: bool = Field(..., description="If you see errors, return True, otherwise, return False")
            feedback: str = Field(..., description="Description of feedback to fix the error following the API documentation of what to fix. In case you don't see any errors (ignore warnings!), return 'Not Applicable'.")
            
        feedback_gen_string = f"""
        You are supposed to go through the stdout and respond whether there are 
        any errors or not. If there are errors, generate feedback necessary to correct 
        code using either API information for errors or standard python coding to fix
        the errros that exist in the standard output. DO NOT EDIT THE CODE. Generate text
        feedback of what SHOULD be done to fix the code.
        """

        self.sys_msg_feedback = SystemMessage(feedback_gen_string + self.api_desc)
        self.feedback_llm = init_chat_model(model=self.model_name,
                                            api_key=api_key,
                                            temperature=temperature_debug)
        
        self.feedback_llm = self.feedback_llm.with_structured_output(Feedback)

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

        message = HumanMessage(content=[
            {"type": "text", "text": script}])
        

        # Initial Code Refinement for any hallucination or coding errors.
        refined_code = self.model_refine.invoke([self.sys_msg_refine, message])

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
            message = HumanMessage(content=[
            {"type": "text", "text": prompt}])

            feedback = self.feedback_llm.invoke([self.sys_msg_feedback, message])
            print(f"Run {i} was executed with result of errors being {feedback.errors}")
            if result and not feedback.errors:
                print("SUCCESS! A full run was executed to completion!")
                break
            
            debug_prompt = f"Previous Script: {refined_code}.\nFeedback: {feedback.feedback}"
            message = HumanMessage(content=[
            {"type": "text", "text": debug_prompt}])

            refined_code = self.model_debugger([self.sys_msg_debug, debug_prompt])

            output_trace, result = self.exec(script_file=script_file_name, script_code=refined_code)

        return refined_code
        
class CodeOptimizerLLM():
    def __init__(self):
        pass

class DebugAndOptimizeAgent():
    def __init__(self):
        pass

