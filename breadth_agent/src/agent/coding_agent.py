from openai import OpenAI
from sceneprogllm import LLM
import glob
import os 
import numpy as np

class DescriptionEncoder():
    def __init__(self, model: str = "text-embedding-3-small",
                    dimensions: int = 256,
                    encoding_format: str ="float"
                 ):
        
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format

        self.client = OpenAI()

    def encode_desc(self, query: str):
        clean_query = query.lstrip().rstrip().replace("\n", "")

        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=clean_query,
            dimensions=256,
            encoding_format="float"
            )
        
        return np.array(response.data[0].embedding)
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0  # Handle cases where one or both vectors are zero vectors

        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

        return cosine_similarity
    
    def compare_encoding(self, query_embed: np.ndarray, db_embed: list[np.ndarray], k_choices: int = 2):
        score_list = []
        for vec in db_embed:
            score = self.cosine_similarity(query_embed, vec)
            score_list.append((score, vec))

        sorted_scores = sorted(score_list)
        print(sorted_scores)
        chosen_vecs = [desc[1] for desc in sorted_scores[:k_choices]]

        return chosen_vecs
        

    def context_db_reader(self, file_path: str, file_path_context: str):
        file_to_read = open(file_path, 'r')
        file_lines = file_to_read.readlines()
        desc_list = []

        embed_vec_list = []
        embed_vec_to_desc = {}
        desc_to_script = {}
        
        for i in range(0, len(file_lines), 2):
            embed_vec = file_lines[i]
            embed_vec = np.array(list(map(float, embed_vec.replace('[', '').replace(']', '').split(','))))
            desc = file_lines[i+1]
            desc_list.append(desc)

            embed_vec_list.append(embed_vec)
            embed_vec_to_desc[embed_vec.tobytes()] = desc

        context_scripts = open(file_path_context, 'r')
        content = context_scripts.read()

        split_db = content.split("===$&$===")

        for i in range(1, len(split_db) - 1):
            context_script = split_db[i].split("# ==#$#==")
            #descs = context_script[0].lstrip().rstrip().replace('"""','').replace("\n", "")
            descs = desc_list[i-1]
            script = context_script[1]

            desc_to_script[descs] = script

        return embed_vec_list, embed_vec_to_desc, desc_to_script

class CodeAgent():
    def __init__(self, 
                 full_length_path: str,
                 embedding_database_path: str,
                 model: str | None = None, 
                 instruction_path: str | None = None,
                 api_directory: str | None = None,
                 k_choices: int = 2,
                 response_format: str = "code",
                 use_cache: bool = False,
                 temperature: float = 0.8):
        
        self.encoder_helper = DescriptionEncoder()

        if instruction_path is None:
            examples = ""
        else:
            context = open(instruction_path, 'r')
            examples = context.read()
            context.close()

        default_desc = """
You are an assisstant with focus on computer vision problems. Please write code for the computer vision problems you see that 
contain a problem statement of the approach to use, description of the scene to inform you what modules or tools to use from
the provided api, and a list of sub-problems with solutions and description to follow. Solve the combination of the sub-problems 
into one solution script. The code must be written in python utilizing the python API provided. Do not utilize the visualizer module,
I will add visualization myself later, so ignore the Visualizer module when constructing the code.
"""

        if api_directory is None:
            self.api_files = []
        else:
            self.api_files = sorted(glob.glob(api_directory + "\\*"))

        api_desc = ""

        for file in self.api_files:
            context = open(file, 'r')
            examples = context.read()
            context.close()
            api_desc += examples + "\n"

        if model is None:
            self.model = "gpt-4o-mini"
        else:
            self.model = model

        self.system_desc = default_desc + examples + "\n" + api_desc

        self.code_generator = LLM("llm_code_generator", 
                                    system_desc = self.system_desc,
                                    response_format=response_format,
                                    use_cache = use_cache,
                                    model_name = self.model,
                                    temperature = temperature)
        
        self.k_choices = k_choices
        
        embed_vec_list, embed_vec_to_desc, desc_to_script = self.encoder_helper.context_db_reader(embedding_database_path, 
                                                                                             full_length_path)
        
        self.embed_vec_list = embed_vec_list
        self.embed_db = embed_vec_to_desc
        self.context_db = desc_to_script


    def __call__(self, enhanced_prompt: str, problem_statement: str):

        ps_embedding = self.encoder_helper.encode_desc(problem_statement)

        chosen_vecs = self.encoder_helper.compare_encoding(ps_embedding, self.embed_vec_list)

        context_builder = ''
        for vecs in chosen_vecs:
            desc = self.embed_db[vecs.tobytes()]
            context = self.context_db[desc]

            context_builder += context + "\n"

        # print(context_builder)
        full_prompt = enhanced_prompt + '\nCoding Examples:\n' + context_builder

        code = self.code_generator(full_prompt)

        return code
    