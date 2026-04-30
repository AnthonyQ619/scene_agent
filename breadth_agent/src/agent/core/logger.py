import os
import random
import numpy as np
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, desc, log_dir="results"):
        self.desc = desc
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Data containers
        self.initial_prompt = []
        self.enhanced_prompt = None                   
        self.initial_workflow = []          
        self.query_image = None

        self.codes1 = []
        self.plans1 = []
        self.outputs1 = []  
        self.codes2 = []
        self.plans2 = []
        self.outputs2 = [] 
        self.codes3 = []
        self.plans3 = []
        self.outputs3 = []                      

        self.best_workflow1 = []
        self.best_workflow2 = []
        self.best_workflow3 = []
        # self.best_code1 = None
        # self.best_plan1 = None
        # self.best_output1 = None              
        # self.best_code2 = None
        # self.best_plan2 = None
        # self.best_output2 = None  
        # self.best_code3 = None
        # self.best_plan3 = None
        # self.best_output3 = None  

        self.final_code = None
        self.final_plan = None
        self.final_output = None

        self.id = random.randint(10000, 99999)

    def _read_image(self, path):
        """Read and return image as numpy array (raises if missing)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Render not found: {path}")
        return plt.imread(path)

    # --- Logging methods ---
    def add_initial_prompt(self, image_path, reconstruction_type, gpu_mem, calibration_path = None, scene_img=None):
        image = self._read_image(scene_img) if scene_img else None
        self.initial_prompt.append((image_path, reconstruction_type, gpu_mem, calibration_path))
        self.query_image = image

    def add_enhanced_prompt(self, enhanced_prompt):
        self.enhanced_prompt = enhanced_prompt

    def add_initial_workflow(self, initial_plan, initial_code, initial_output):
        self.initial_workflow.append((initial_plan, initial_code, initial_output))

    def add_generated_codes_batch(self, current_batch, index):
        if index == 1: 
            for idx, batch in enumerate(current_batch):
                pl, prog, output = batch[:3]
                self.codes1.append(pl)
                self.plans1.append(prog)
                self.outputs1.append(output)
        elif index == 2:
            for idx, batch in enumerate(current_batch):
                pl, prog, output = batch[:3]
                self.codes2.append(pl)
                self.plans2.append(prog)
                self.outputs2.append(output)
        elif index == 3:
            for idx, batch in enumerate(current_batch):
                pl, prog, output = batch[:3]
                self.codes3.append(pl)
                self.plans3.append(prog)
                self.outputs3.append(output)

    def add_best_code(self, plan, program, output, index):
        if index == 1: 
            # self.best_plan1 = plan
            # self.best_code1 = program
            # self.best_output1 = output
            self.best_workflow1.append((plan, program, output))
        elif index == 2: 
            # self.best_plan2 = plan
            # self.best_code2 = program
            # self.best_output2 = output
            self.best_workflow2.append((plan, program, output))
        elif index == 3: 
            # self.best_plan3 = plan
            # self.best_code3 = program
            # self.best_output3 = output
            self.best_workflow3.append((plan, program, output))

    def add_final_code(self, plan, program, output):
        self.final_code = program
        self.final_plan = plan
        self.final_output = output

    # --- Save / serialize ---

    def save(self):
        """Save all logs (with image arrays) to a single compressed .npz file."""
        out_path = os.path.join(self.log_dir, f"{self.desc}_log.npz")
        np.savez_compressed(
            out_path,
            initial_prompt = np.array(self.initial_prompt, dtype=object),
            enhanced_prompt = np.array(self.enhanced_prompt, dtype=object),                  
            initial_workflow = np.array(self.initial_workflow, dtype=object),         
            query_image = np.array(self.query_image, dtype=object),
            codes1 = np.array(self.codes1, dtype=object),
            plans1 = np.array(self.plans1, dtype=object),
            outputs1 = np.array(self.outputs1, dtype=object),  
            codes2 = np.array(self.codes2, dtype=object),
            plans2 = np.array(self.plans2, dtype=object),
            outputs2 = np.array(self.outputs2, dtype=object),
            codes3 = np.array(self.codes3, dtype=object),
            plans3 = np.array(self.plans3, dtype=object),
            outputs3 = np.array(self.outputs3, dtype=object),                     
            best_workflow1 = np.array(self.best_workflow1, dtype=object), 
            best_workflow2 = np.array(self.best_workflow2, dtype=object),
            best_workflow3 = np.array(self.best_workflow3, dtype=object),
            final_code = np.array(self.final_code, dtype=object),
            final_plan = np.array(self.final_plan, dtype=object),
            final_output = np.array(self.final_output, dtype=object),
            # prompt=np.array(self.prompt, dtype=object),
            # contexts=np.array(self.contexts, dtype=object),
            # initial_procedures=np.array(self.initial_procedures, dtype=object),
            # best_initial_procedure=np.array(self.best_initial_procedure, dtype=object),
            # initial_code_doc=np.array(self.initial_code_doc, dtype=object),
            # codes1=np.array(self.codes1, dtype=object),
            # best_code1=np.array(self.best_code1, dtype=object),
            # codes2=np.array(self.codes2, dtype=object),
            # best_code2=np.array(self.best_code2, dtype=object),
            # codes3=np.array(self.codes3, dtype=object),
            # best_code3=np.array(self.best_code3, dtype=object),
            # codes4=np.array(self.codes4, dtype=object),
            # best_code4=np.array(self.best_code4, dtype=object),
        )
        print(f"[Logger] Saved logs to {out_path}")