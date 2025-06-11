from openai import OpenAI
import os
import json


# Apply polymorphism
class CodeGenerator:
  def __init__(self, api_key, model_url, system_header=None, system_closer=None):
    self.api_key = api_key
    self.model_url = model_url

    # User input
    if system_header is not None:
      self.code_header = system_header
    else:
      # Generated Code System Description (Default)
      self.code_header = f"""For the following prompt, generate a python script that
      solves the computer vision problem using OpenCV or other open source
      computer vision libraries. If you need to use torch, apply gpu and use cuda
      for more efficient runtimes. If there is a use for machine learning
      models, ensure the use of pre-trained models that do not require us
      to download the weights through a file and access it in the code.

      """
    if system_closer is not None:
      self.code_closer = system_closer
    else:
      # Generated Code System Description (Default)
      self.code_closer = f"""
      Finally, output the result into a file if the problem calls for it.
      """

  def generate_code(self, prompt): # To be overwritten
    pass


class ChatGPT(CodeGenerator):
  def __init__(self, api_key, model, model_url=None,
               use_cache=False, response_type="code", k_ref=0,
               system_header=None, system_closer=None):

    self.system_header = system_header
    self.system_closer = system_closer

    super().__init__(api_key, model_url, system_header = self.system_header,
                     system_closer = self.system_closer)

    self.use_cache = use_cache
    self.response_type = response_type

    if k_ref > 2:
      self.temperature = 0.7
    else:
      self.temperature = 0.0

    self.client = OpenAI(api_key=self.api_key)
    self.model_name = model

    self.instruction = self.code_header + self.code_closer


  def generate_code(self, prompt):
    if self.model_name.lower() == "o3-mini-2025-01-31":
      response = self.client.responses.create(
        model=self.model_name,
        instructions=self.instruction,
        input=prompt)
      response = response.model_dump()['output'][1]['content'][0]['text']
    else:
      if self.instruction is not None:
        response = self.client.responses.create(
          model=self.model_name,
          instructions=self.instruction,
          input=prompt,
          temperature=self.temperature)
      else:
        response = self.client.responses.create(
          model=self.model_name,
          input=prompt,
          temperature=self.temperature)

    # if self.response_type == "code":
    #   script = script_extraction_helper_python(response)

    #   return script
    # else:
    return response

class DeepSeek(CodeGenerator):
  def __init__(self, api_key, model_url):
    super().__init__(api_key, model_url)

  def generate_code(self, prompt):
    pass

class LLM(CodeGenerator):
  def __init__(self, model_name, model_type = None):
    self.model_name = model_name

    self.k = 0
    self.default_system_desc = f"""
    First identify the errors and then respond with the corrected code.
    You should also pay attention to the exceptions raised while running
    the code and find ways to fix them. You are not supposed to change placement
    values or settings in the code, but only watch out for reasons due to which
    the code may crash! Also, you don't have to worry about importing additional
    modules. They are already imported for you. In case the code includes
    imports, make sure that debugged code includes the same imports.
    Make sure that the debugged code does what the original code was intended
    to do and that it is not doing anything extra. The only thing you have to
    do is to fix the errors.
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

    code:
    import numpy as np
    a = np.array([1,2,3])
    b = np.array([4,5,6],[7,8,9])

    debugged code:
    import numpy as np
    a = np.array([1,2,3])
    b = np.array([[4,5,6],[7,8,9]])

    """

    if (model_type == None) or (model_type.lower() == "generator"): # Assume Generator Case
      system_header = None # None contains the defalt code generator script
      system_closer = None # None contains the defalt code generator script
    elif model_type.lower() == "debugger":
      self.k = 3
      # Debugging Code System Description
      system_header = f"""
    You should go through the code as well as the traceback
    and find the errors including those caused by wrong use of the API. Then
    you must respond with the corrected code.

    """
      system_closer = self.default_system_desc
    elif model_type.lower() == "refiner":
      # Refinement Code System Description
      system_header = f"""You should go through the code and find the errors
    including those caused by wrong use of the API. Then you must respond
    with the corrected code. Only add the code that is necessary to fix
    the errors. Don't add any other code.

    """
      system_closer = self.default_system_desc
    elif model_type.lower() == "feedback":
      system_header = f"""You should go through the code and find the errors
    including those caused by wrong use of the API. Then you must generate a
    response with the necesary feedback to correct the code.
    Only use words in your response and don't add any other code.

    """
      system_closer = f"""You should also pay attention to the exceptions raised while running
    the code and find ways to fix them. You are not supposed to change placement
    values or settings in the code, but only watch out for reasons due to which
    the code may crash! Also, you don't have to worry about importing additional
    modules. They are already imported for you.In case the code includes
    imports, make sure that debugged code includes the same imports.
    Make sure that the debugged code does what the original code was intended
    to do and that it is not doing anything extra. The only thing you have to
    do is to fix the errors.
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
    else:
      print("ERROR: must choose one of the following - debugger, refiner, feedback, or generator")
      return None


    #self.refine_instruction = self.refine_header + self.default_system_desc

    if self.model_name.lower() == "chatgpt":
      self.model = ChatGPT(open_ai_key, "gpt-4o-mini-2024-07-18", k_ref = self.k,
      #self.model = ChatGPT(open_ai_key, "o3-mini-2025-01-31", k_ref = self.k,
                           system_header=system_header,
                           system_closer=system_closer)
    elif self.model_name.lower() == "deepseek":
      self.model = DeepSeek()

  def generate_code(self, prompt):
    return self.model.generate_code(prompt)