from core.compiler import Compiler
from core.generator import Generator
from core.promptenhancer import PromptEnhancerLLM
from sceneprogllm import LLM

class AutoSFM:
    def __init__(self, model_name, api_directory, instruction_path, reasoning_effort):
        self.compiler = Compiler(model=model_name, api_directory=api_directory, instruction_path=instruction_path, reasoning_effort=reasoning_effort)
        self.generator = Generator(model=model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.enhancer = PromptEnhancerLLM(model=model_name, instruction_path=instruction_path, reasoning_effort=reasoning_effort)

        self.evaluator_llm = LLM(
            system_desc="""
You are an expert Structure from Motion workflow plans.
Given the image of the scene and reconstruction guidelines we wish to follow (prompt) and several step-by-step node procedures, your task is to:
	1.	Analyze each SfM plan in detail — what modules are called, parameters used, and if we invoke the correct sub-modules for the reconstruction type.
	2.	Compare how accurately each sub-module that is invoked best fits the given scene we wish to reconstruct.
	3.	Choose the best procedure and explain why — citing specific SfM planning logic, module selection, and why parameterization of tools make sense.

Judge using these key aspects:
	•	Choice of sub-modules accurately coincide with the image of the scene and best use-cases.
	•	Sub-module selection fits within the system constraints of the user prompt.
	•	Reconstruction type is followed precisely, and our last sub-module invoked directly represents the prompt (Pose, Sparse, or Dense).

You are to provide me the index (starting from 1) of the best plan amongst those provided to you, so that I can pick it easily.
Your response should be a single integer indicating the best plan index, without any additional commentary.
""",
            response_format="json",
            response_params={"best_plan_index":"int"},
            model_name=model,
            reasoning_effort=reasoning_effort
        )
    def evaluate_plans(self, plans):


        return plans[0][0], plans[0][1], plans[0][2] ## For now, just return the first plan. We can implement a more complex evaluation strategy later.

    def run(self, prompt):
        print("Enhancing prompt...")
        new_query = self.enhancer(prompt)
        new_query = f"""
{new_query["statement"]}
{new_query["reconstruction"]}
{new_query["calibration"]}
{new_query["memory"]}
Use Image Path in Code: {prompt['images']}
Use Calibration Path in Code: {prompt['calibration']}
"""
        
        output = None
        print("Generating initial plan...")
        plan = self.generator(new_query, prompt['images']) ## Initial plan
        breakpoint()
        print("Running initial plan...")
        program, output = self.compiler(plan) ## initial feedback
        breakpoint()
        """
        print("Generating first set of multiple plans...")
        plans = self.generator.forward(new_query, prompt['images'], feedback=output, self_evaluate=False) ## Generate multiple plans
        best_plan = None
        best_output = None
        for i in range(3):
            if best_plan is not None:
                current_batch = [(best_plan, best_output)]
            else:
                current_batch= []
            print(f"Evaluating plans for iteration {i + 1}...")
            for j in range(len(plans)):
                program, output = self.compiler(plans[j]) ## Get feedback for each plan
                current_batch.append((plans[j], program, output))
            
            print(f"Selecting best plan for iteration {i + 1}...")
            best_plan, best_program, best_output = self.evaluate_plans(current_batch) 
            self.generator.plan = best_plan ## Set the best plan for the next iteration
            if i == 2:
                break
            print(f"Generating next set of plans for iteration {i + 1}...")
            plans = self.generator.forward(new_query, prompt['images'], feedback=best_output, self_evaluate=False)
                
        print("Done with iterations. Returning best plan and output.")
        return best_plan, best_program, best_output"""
        return program, output