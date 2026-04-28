from core.compiler import Compiler
from core.generator import Generator
from core.promptenhancer import PromptEnhancerLLM

class AutoSFM:
    def __init__(self, model_name, api_directory, instruction_path, reasoning_effort):
        self.compiler = Compiler(model=model_name, api_directory=api_directory, instruction_path=instruction_path, reasoning_effort=reasoning_effort)
        self.generator = Generator(model=model_name, api_directory=api_directory, reasoning_effort=reasoning_effort)
        self.enhancer = PromptEnhancerLLM(model=model_name, instruction_path=instruction_path, reasoning_effort=reasoning_effort)
    
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
        print("Running initial plan...")
        program, output = self.compiler(plan) ## initial feedback
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
        return best_plan, best_program, best_output