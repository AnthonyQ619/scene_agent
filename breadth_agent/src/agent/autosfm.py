from core.compiler import Compiler
from core.generator import Generator
from core.promptenhancer import PromptEnhancerLLM
from sceneprogllm import LLM
from pathlib import Path
import os

class AutoSFM:
    def __init__(self, model_name, api_directory, instruction_path, reasoning_effort):
        self.compiler = Compiler(model=model_name, api_directory=api_directory, instruction_path=instruction_path, reasoning_effort=reasoning_effort)
        self.generator = Generator(model=model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.enhancer = PromptEnhancerLLM(model=model_name, instruction_path=instruction_path, reasoning_effort=reasoning_effort)

        self.evaluator_llm = LLM(
            system_desc=f"""
You are an expert on Structure from Motion workflow planning and optimization.
Given the image of the scene, reconstruction guidelines and the generated plan with corresponding metrics/feedback for improvement:
	1.	Analyze each SfM plan in detail — what modules are called, parameters used, and if we invoke the correct sub-modules for the reconstruction type.
	2.	Compare how successful each sub-module is performing for the given plan. Verify if it is successful or can be improved upon.
	3.	Choose the best plan with the best performing metrics.
    4.  If all plans only have feedback due to performance errors, select any of the plans available when no metris are provided

Judge using these key aspects:
	•	Choice of sub-modules accurately coincide with the image of the scene and best use-cases.
	•	Sub-module performance from the metrics follow accordingly given context examples from the prompt.
	•	Reconstruction type is followed precisely, and our last sub-module invoked directly represents the prompt (Pose, Sparse, or Dense).

Following are the guidelines of what each metric means:
{self.generator.metric_prompt}

You are to provide me the index (starting from 1) of the best plan amongst those provided to you, so that I can pick it easily.
Your response should be a single integer indicating the best plan index, without any additional commentary.
""",
            response_format="json",
            response_params={"best_plan_index":"int"},
            model_name=model_name,
            reasoning_effort=reasoning_effort
        )

        evaluation_context = ""

        self.CWD = str(Path(__file__).resolve().parents[0])

        full_context_plan_metric_f = os.path.join(self.CWD, 'agent_details', 'optimize_context', 'script_metrics.txt')
        full_context_file_proc_f = os.path.join(self.CWD, 'agent_details', 'script_context', 'process.txt')

        # Read Process/Metric Files
        metric_context = full_context_plan_metric_f.read()
        process_context = full_context_file_proc_f.read()

        ## Split Context
        metric_context = metric_context.split("=%$%=")
        process_context = process_context.split("==$#$==")

        for i in range(len(process_context) - 1):
            plan = process_context[i]
            metrics = metric_context[i]
            evaluation_context += f"Example {i + 1}:\n" + plan + "\n" + metrics + "\n"

        self.p_m_context = evaluation_context

    def evaluate_plans(self, plans, user_query):
        enhanced_prompt = f"""
Given the user query:
{user_query}
And the attached image of the Scene to reconstruct, we want to generate a plan that is optimal for the 
given scene. 
Here are a few examples of plans and their resulting metrics:
{self.p_m_context}

You are now given a set of plans and their resulting metrics from generating a SfM plan to the given scene:\n
"""
        for idx, pl, prog, output in enumerate(plans):
            enhanced_prompt += f"\nPlan {idx+1}:\n{pl}\nResult {idx+1}:\n{output}\n"
        enhanced_prompt += "\nSelect the plan that generates the best set of metrics, if no metrics are provided (All plans failed to execute), choose any plan as a result"

        response = self.evaluator_llm(enhanced_prompt, image_paths=self.generator.new_query_img_path)["best_plan_index"]
        best_index = int(response) - 1  # Convert to 0-based index
        return plans[best_index][0], plans[best_index][1], plans[best_index][2] ## For now, just return the first plan. We can implement a more complex evaluation strategy later.

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