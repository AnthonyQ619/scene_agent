from sceneprogllm import LLM, SceneProgTemplate
import glob
from pydantic import BaseModel, Field
import os
from pathlib import Path

class Debugger:
    def __init__(self,
                 executor,
                 template: SceneProgTemplate, 
                 model="gpt-5", 
                 reasoning_effort="minimal", 
                 temperature_debug: float = 0.8,
                 temperature_refine: float = 0.8,
                 api_path=None,
                 max_attempts = 5, 
                 visualize=False):
        # Logistical Variables Set
        self.template = template
        self.visualize = visualize
        self.MAX_ATTEMPTS = max_attempts
        self.exec = executor

        # Get API Documentation
        if api_path is None:
            self.api_files = []
        else:
            self.api_files = sorted(glob.glob(api_path + "\\*"))

        api_desc = ""
        for file in self.api_files:
            context = open(file, 'r')
            examples = context.read()
            context.close()
            api_desc += examples + "\n"

        # Set up the LLM Modules

        ## Setup Refiner Module
        self.code_refiner = LLM(
            system_desc=self.get_code_refiner_prompt(api_desc),
            response_format="code",
            model_name=model,
            reasoning_effort=reasoning_effort,
            temperature = temperature_refine 
        )

        ## Setup Debugger Module
        self.code_debugger = LLM(
            system_desc=self.get_code_debugger_prompt(api_desc),
            response_format="code",
            model_name=model,
            reasoning_effort=reasoning_effort,
            temperature = temperature_debug 
        )

        ## Setup Feedback Module
        feedback_gen_string = f"""
        API Documentation:
        {api_desc}

        Instruction:
        You are supposed to go through the stdout and respond whether there are 
        any errors or not. If there are errors, generate feedback necessary to correct 
        code using either API information for errors or standard python coding to fix
        the errors that exist in the standard output.
        
        Ignore all warnings, do not report feedback on any warnings or COLMAP Linear Solver Failures. 
        These are not errors, only respond to errors when code execution fails, so ignore all warnings.
        Simply ignore this warning as it does NOT havey any effect on the code as it's not important. 
        A specific case is "Linear solver failure. Failed to compute a step: Eigen failure" IGNORE THIS WARNING. 
        I REPEAT, DO NOT ACKNOWLEDGE THIS WARNING. 
        
        DO NOT EDIT THE CODE. Generate text
        feedback of what SHOULD be done to fix the code.
        """

        self.code_feedback = LLM(system_desc = feedback_gen_string,
                                response_format = "pydantic",
                                model_name = model,
                                reasoning_effort = reasoning_effort)

    def get_code_refiner_prompt(self, api_desc):
        try:
            return api_desc + "\n" + self.template.get_section("<cr>", "</cr>")
        except ValueError:
            return (
                "You are an expert in debugging Python and Blender code. "
                "Go through the following code and check for any errors. "
                "Respond with the corrected code only. Don't add any explanations or extra text."
            )
    
    def get_code_debugger_prompt(self, api_desc):
        try:
            return api_desc + "\n" + self.template.get_section("<tr>", "</tr>")
        except ValueError:
            return (
                "You are an expert in debugging Python and Blender code. "
                "Given the code and the stdout after execution, fix the errors. "
                "Return only the corrected code. Don't add any explanations or extra text."
            )

    def __call__(self, code: str):
        class Feedback(BaseModel):
            errors: bool = Field(..., description="If you see errors, return True, otherwise, return False")
            feedback: str = Field(..., description="Description of feedback to fix the error following the API documentation of what to fix. In case you don't see any errors (ignore warnings!), return 'Not Applicable'.")

        # # Initial Run
        # script_path = os.path.join(self.script_dir, script_file_name)

        # # Read Code from Script
        # f_script = open(script_path, 'r')
        # script = f_script.read()
        # f_script.close()


        # Initial Code Refinement for any hallucination or coding errors.
        refined_code = self.code_refiner(code)

        # Initial Code Running Stage
        output_trace, result = self.exec(script_code=refined_code)

        print("OUTPUT")
        print(output_trace)

        print(result)
        print("END OUTPUT")

        # prompt = f"Input: \n {script}.\nErrors: {output_trace}."

        # feedback = self.feedback_gen(prompt)
        # print(feedback)
        for i in range(self.MAX_ATTEMPTS): # Maximum number of attempts
            prompt = f"Input: \n {refined_code}.\nErrors: {output_trace}."

            feedback = self.code_feedback(prompt, pydantic_object=Feedback)
            print(f"Run {i} was executed with result of errors being {feedback.errors}")
            if result and not feedback.errors:
                print("SUCCESS! A full run was executed to completion!")
                break
            
            debug_prompt = f"Previous Script: {refined_code}.\nFeedback: {feedback.feedback}"

            refined_code = self.code_debugger(debug_prompt)

            # refined_code = self.code_refiner(refined_code)

            output_trace, result = self.exec(script_code=refined_code)

        return refined_code