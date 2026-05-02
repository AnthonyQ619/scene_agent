from core.compiler import Compiler
from core.generator import Generator
from core.promptenhancer import PromptEnhancerLLM
from core.logger import Logger
from sceneprogllm import LLM
from pathlib import Path
import ast
import os

class AutoSFM:
    def __init__(self, model_name, api_directory, instruction_path, reasoning_effort, logger):
        self.logger = logger
        self.log_dir = logger.log_dir

        self.compiler = Compiler(model=model_name, api_directory=api_directory, instruction_path=instruction_path, reasoning_effort=reasoning_effort, log_dir=self.log_dir)
        self.generator = Generator(model=model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.enhancer = PromptEnhancerLLM(model=model_name, instruction_path=instruction_path, reasoning_effort=reasoning_effort)

        # self.logger = logger
        # self.log_dir = logger.log_dir

        self.evaluator_llm = LLM(
            system_desc=f"""
You are an expert on Structure from Motion workflow planning and optimization.
Given the image of the scene, reconstruction guidelines and the generated plan with corresponding metrics/feedback for improvement:
	1.	Analyze each SfM plan in detail — what modules are called, parameters used, and if we invoke the correct sub-modules for the reconstruction type.
	2.	Compare how successful each sub-module is performing for the given plan. Verify if it is successful or can be improved upon.
	3.	Choose the best plan with the best performing metrics.
    4.  If all plans only have feedback due to performance errors, select any of the plans available when no metris are provided

Judge using these key aspects:
	•	Keep in mind, the context provided are just baselines, when aiming to choose the best work flow consider the following:
    	•	Lower reprojection error across Pose Estimation and Global Optimization when Relevant
    	•	Number of 3D points in reconstruction (The more 3D points the better, especially when final reprojection errors are comparable and the 
            scene with the higher 3D points have a slightly higher reprojection error - within 0.20 pixel error.)
    	•	Whether Global optimization Converges (Prioritize other metrics, such as Feature Matches/Tracks, 3D Points, etc higher, but still consider
            Convergence when multiple plans are performing similarly - Essentially use it as a tie breaker)
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

        full_context_plan_metric_dir = os.path.join(self.CWD, 'agent_details', 'optimize_context', 'script_metrics.txt')
        full_context_file_proc_dir = os.path.join(self.CWD, 'agent_details', 'script_context', 'process.txt')
        full_context_plan_metric_f = open(full_context_plan_metric_dir, 'r')
        full_context_file_proc_f = open(full_context_file_proc_dir, 'r')

        # Read Process/Metric Files
        metric_context = full_context_plan_metric_f.read()
        process_context = full_context_file_proc_f.read()

        ## Split Context
        metric_context = metric_context.split("=%$%=")
        process_context = process_context.split("==$#$==")

        for i in range(len(process_context)):
            plan = process_context[i]
            metrics = metric_context[i]
            evaluation_context += f"Example {i + 1}:\n" + plan + "\n" + metrics + "\n"

        full_context_plan_metric_f.close()
        full_context_file_proc_f.close()
        # breakpoint()
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
        for idx, batch in enumerate(plans):
            pl, prog, output = batch[:3]
            enhanced_prompt += f"\nPlan {idx+1}:\n{pl}\nResult {idx+1}:\n{output}\n"
        enhanced_prompt += "\nSelect the plan that generates the best set of metrics, if no metrics are provided (All plans failed to execute), choose any plan as a result"

        response = self.evaluator_llm(enhanced_prompt, image_paths=[self.generator.new_query_img_path])["best_plan_index"]
        best_index = int(response) - 1  # Convert to 0-based index
        return plans[best_index][0], plans[best_index][1], plans[best_index][2], plans[best_index][3] #Plan, Program, Output, Program_id

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
        self.logger.add_initial_prompt(prompt['images'], prompt['recon_type'], prompt['gpu_mem'], prompt['calibration'], self.generator.new_query_img_path)
        self.logger.add_enhanced_prompt(new_query)
        # breakpoint()
        print("Running initial plan...")
        program, output, prog_id = self.compiler(plan) ## initial feedback # Aplly script_id as output
        self.logger.add_initial_workflow(plan, program, output, prog_id)
        # breakpoint()
        
        print("Generating first set of multiple plans...")
        plans = self.generator.forward(new_query, prompt['images'], feedback=output, self_evaluate=False) ## Generate multiple plans
        best_plan = None
        best_output = None
        for i in range(3): 
            if best_plan is not None:
                current_batch = [(best_plan, best_program, best_output, best_prog_id)]
            else:
                current_batch= []
            print(f"Evaluating plans for iteration {i + 1}...")
            for j in range(len(plans)):
                program, output, prog_id = self.compiler(plans[j]) ## Get feedback for each plan
                current_batch.append((plans[j], program, output, prog_id))
            self.logger.add_generated_codes_batch(current_batch, i + 1)
            # breakpoint()
            print(f"Selecting best plan for iteration {i + 1}...")
            best_plan, best_program, best_output, best_prog_id = self.evaluate_plans(current_batch, new_query) 
            self.logger.add_best_code(best_plan, best_program, best_output, best_prog_id , i+1)
            # breakpoint()
            self.generator.plan = best_plan ## Set the best plan for the next iteration
            if i == 2:
                break
            print(f"Generating next set of plans for iteration {i + 1}...")
            plans = self.generator.forward(new_query, prompt['images'], feedback=best_output, self_evaluate=False)
            # breakpoint()

        print("Done with iterations. Returning best plan and output.")
        print("Running Best Plan/Output in Full!")
        full_best_program = remove_keyword_from_first_call(
            best_program,
            call_name="SfMScene",
            keyword_name="max_images",
        )
        self.logger.add_final_code(best_plan, full_best_program, best_output, best_prog_id)
        # breakpoint()
        self.compiler.exec(full_best_program)
        self.logger.save()
        return best_plan, best_program, best_output
        # return program, output

# Helper function to clean code before final run!
def remove_keyword_from_first_call(
    script: str,
    call_name: str,
    keyword_name: str,
) -> str:
    """
    Removes one keyword argument from the first call to `call_name`.

    Example:
        remove_keyword_from_first_call(
            script,
            call_name="SfMScene",
            keyword_name="max_images"
        )

    Removes:
        max_images=20

    while preserving the rest of the script, including comments.
    """

    tree = ast.parse(script)

    # Convert line/column offsets to absolute string offsets
    line_starts = []
    offset = 0
    for line in script.splitlines(keepends=True):
        line_starts.append(offset)
        offset += len(line)

    def abs_offset(lineno: int, col: int) -> int:
        return line_starts[lineno - 1] + col

    target_keyword = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        # Match SfMScene(...)
        if isinstance(node.func, ast.Name) and node.func.id == call_name:
            for kw in node.keywords:
                if kw.arg == keyword_name:
                    target_keyword = kw
                    break

        if target_keyword is not None:
            break

    if target_keyword is None:
        return script

    start = abs_offset(target_keyword.lineno, target_keyword.col_offset)
    end = abs_offset(target_keyword.end_lineno, target_keyword.end_col_offset)

    # Extend removal to include a following comma, if present
    i = end
    while i < len(script) and script[i].isspace():
        i += 1

    if i < len(script) and script[i] == ",":
        end = i + 1
    else:
        # Otherwise remove the preceding comma
        j = start - 1
        while j >= 0 and script[j].isspace():
            j -= 1

        if j >= 0 and script[j] == ",":
            start = j

    return script[:start] + script[end:]