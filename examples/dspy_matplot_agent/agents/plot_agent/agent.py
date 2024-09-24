import os
import re


from .prompt import ERROR_PROMPT, INITIAL_SYSTEM_PROMPT, INITIAL_USER_PROMPT, VIS_SYSTEM_PROMPT, VIS_USER_PROMPT, ZERO_SHOT_COT_PROMPT
from agents.openai_chatComplete import completion_with_backoff
from agents.utils import fill_in_placeholders, get_error_message, is_run_code_success, run_code
from agents.utils import print_filesys_struture
from agents.utils import change_directory


class PlotAgent():


    def __init__(self, config, query, data_information=None):
        self.chat_history = []
        self.workspace = config['workspace']
        self.query = query
        self.data_information = data_information

    def generate(self, user_prompt, model_type, query_type, file_name):

        workspace_structure = print_filesys_struture(self.workspace)
        
        information = {
            'workspace_structure': workspace_structure,
            'file_name': file_name,
            'query': user_prompt
        }

        if query_type == 'initial':
            messages = []
            messages.append({"role": "system", "content": fill_in_placeholders(INITIAL_SYSTEM_PROMPT, information)})
            messages.append({"role": "user", "content": fill_in_placeholders(INITIAL_USER_PROMPT, information)})
        else:
            messages = []
            messages.append({"role": "system", "content": fill_in_placeholders(VIS_SYSTEM_PROMPT, information)})
            messages.append({"role": "user", "content": fill_in_placeholders(VIS_USER_PROMPT, information)})

        self.chat_history = self.chat_history + messages
        return completion_with_backoff(messages, model_type)

    def get_code(self, response):

        all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)


        all_code_blocks = all_python_code_blocks_pattern.findall(response)
        all_code_blocks_combined = '\n'.join(all_code_blocks)
        return all_code_blocks_combined
    
    def get_code2(self, response,file_name):

        all_python_code_blocks_pattern = re.compile(r'```\s*([\s\S]+?)\s*```', re.MULTILINE)


        all_code_blocks = all_python_code_blocks_pattern.findall(response)
        all_code_blocks_combined = '\n'.join(all_code_blocks)
        if all_code_blocks_combined == '':

            response_lines = response.split('\n')
            code_lines = []
            code_start = False
            for line in response_lines:
                if line.find('import') == 0 or code_start:
                    code_lines.append(line)
                    code_start = True
                if code_start and line.find(file_name)!=-1 and line.find('(') !=-1 and line.find(')')!=-1 and line.find('(') < line.find(file_name)< line.find(')'): #要有文件名，同时要有函数调用

                    return '\n'.join(code_lines)
        return all_code_blocks_combined


    def run(self, query, model_type, query_type, file_name):
        try_count = 0
        image_file = file_name
        result = self.generate(query, model_type=model_type, query_type=query_type, file_name=file_name)
        while try_count < 4:
            
            if not isinstance(result, str):  # 如果返回的不是字符串，那么就是出错了
                return 'TOO LONG FOR MODEL', code
            if model_type != 'gpt-4':
                code = self.get_code(result)
                if code.strip() == '':
                    code = self.get_code2(result,image_file) #第二次尝试获得代码
                    if code.strip() == '':
                        code = result  #只能用原始回答
                        if code.strip() == '' and try_count == 0: #有可能是因为没有extend query写好了代码，所以他不写代码
                            code = self.get_code(query)
            else:
                code = self.get_code(result)
            self.chat_history.append({"role": "assistant", "content": result if result.strip() != '' else ''})


            file_name = f'code_action_{model_type}_{query_type}_{try_count}.py'
            with open(os.path.join(self.workspace, file_name), 'w') as f:
                f.write(code)
            error = None
            log = run_code(self.workspace, file_name)

            if is_run_code_success(log):
                if print_filesys_struture(self.workspace).find('.png') == -1:
                    log = log + '\n' + 'No plot generated.'
                    
                    self.chat_history.append({"role": "user", "content": fill_in_placeholders(ERROR_PROMPT,
                                                                                          {'error_message': f'No plot generated. When you complete a plot, remember to save it to a png file. The file name should be """{image_file}""".',
                                                                                           'data_information': self.data_information})})
                    try_count += 1
                    result = completion_with_backoff(self.chat_history, model_type=model_type)


                else:
                    return log, code

            else:
                error = get_error_message(log) if error is None else error
                # TODO error prompt
                self.chat_history.append({"role": "user", "content": fill_in_placeholders(ERROR_PROMPT,
                                                                                          {'error_message': error,
                                                                                           'data_information': self.data_information})})
                try_count += 1
                result = completion_with_backoff(self.chat_history, model_type=model_type)

        return log, ''

    def run_initial(self, model_type, file_name):
        print('========Plot AGENT Expert RUN========')
        self.chat_history = []
        log, code = self.run(self.query, model_type, 'initial', file_name)
        return log, code

    def run_vis(self, model_type, file_name):
        print('========Plot AGENT Novice RUN========')
        self.chat_history = []
        log, code = self.run(self.query, model_type, 'vis_refined', file_name)
        return log, code

    def run_one_time(self, model_type, file_name,query_type='novice',no_sysprompt=False):
        
        print('========Plot AGENT Novice RUN========')
        message = []
        workspace_structure = print_filesys_struture(self.workspace)
        
        information = {
            'workspace_structure': workspace_structure,
            'file_name': file_name,
            'query': self.query
        }
        if no_sysprompt:
            message.append({"role": "system", "content": ''''''})
        message.append({"role": "user", "content": fill_in_placeholders(INITIAL_USER_PROMPT, information)})
        result = completion_with_backoff(message, model_type)
        if model_type != 'gpt-4':
            code = self.get_code(result)
            if code == '':
                code = self.get_code2(result,file_name)
                if code == '':
                    code = result
        else:
            code = self.get_code(result)


        file_name = f'code_action_{model_type}_{query_type}_0.py'
        with open(os.path.join(self.workspace, file_name), 'w') as f:
            f.write(code)
        log = run_code(self.workspace, file_name)
        return log, code
    def run_one_time_zero_shot_COT(self, model_type, file_name,query_type='novice',no_sysprompt=False):
        
        print('========Plot AGENT Novice RUN========')
        message = []
        workspace_structure = print_filesys_struture(self.workspace)
        
        information = {
            'workspace_structure': workspace_structure,
            'file_name': file_name,
            'query': self.query
        }
        message.append({"role": "system", "content": ''''''})
        message.append({"role": "user", "content": fill_in_placeholders(ZERO_SHOT_COT_PROMPT, information)})
        result = completion_with_backoff(message, model_type)
        if model_type != 'gpt-4':
            code = self.get_code(result)
            if code == '':
                code = self.get_code2(result,file_name)
                if code == '':
                    code = result
        else:
            code = self.get_code(result)

        file_name = f'code_action_{model_type}_{query_type}_0.py'
        with open(os.path.join(self.workspace, file_name), 'w') as f:
            f.write(code)
        log = run_code(self.workspace, file_name)
        return log, code


import dspy
from agents.dspy_common import OpenAIModel
from agents.config.openai import openai_kwargs

class PlotCoder(dspy.Signature):
    '''
    You are a cutting-edge super capable code generation LLM. You will be given a natural language query, generate a runnable python code to satisfy all the requirements in the query. You can use any python library you want. 

    If the query requires data manipulation from a csv file, process the data from the csv file and draw the plot in one piece of code.

    In your code, when you complete a plot, remember to save it to a png file with given the 'plot_file_name'.
    Please only return the python code. Wrap it with ```python and ``` to format it properly
    '''
    
    expanded_query = dspy.InputField(format=str)
    plot_file_name = dspy.InputField(format=str)
    
    code = dspy.OutputField(format=str)

class Debugger(dspy.Signature):
    """
    You are a cutting-edge super capable code debugger LLM. You will be given a user query about data visualization, a piece of existing python code for completing the task and an error message associate with this code. 

    Your task is to fix the error. You can use any python library you want. The code should be executable and can at least generate a plot without any error. Please only return the python code. Wrap it with ```python and ``` to format it properly.
    """
    
    query = dspy.InputField(format=str)
    err_code = dspy.InputField(format=str)
    error_message = dspy.InputField(format=str)
    
    code = dspy.OutputField(format=str)

class PlotRefiner(dspy.Signature):
    """
    You are a cutting-edge super capable code generation LLM. You will be given a piece of code and natural language instruction on how to improve it. Base on the given code, generate a runnable python code to satisfy all the requirements in the instruction while retaining the original code's functionality. You can use any python library you want. 

    In your code, when you complete a plot, remember to save it to a png file with given the 'plot_file_name'. Please only return the python code. Wrap it with ```python and ``` to format it properly.
    """
    refine_instruction = dspy.InputField(format=str)
    plot_file_name = dspy.InputField(format=str)
    
    code = dspy.OutputField(format=str)
    
class PlotAgentModule(dspy.Module):
    def __init__(self, coder, debugger, model_type, data_information=None):
        self.chat_history = []
        self.data_information = data_information
        self.coder = coder
        self.debugger = debugger
        self.model_type = model_type
        self.engine = OpenAIModel(
            model=model_type, **openai_kwargs
        )
    
    def get_code(self, response):
        all_python_code_blocks_pattern = re.compile(r'```python\s*([\s\S]+?)\s*```', re.MULTILINE)

        all_code_blocks = all_python_code_blocks_pattern.findall(response)
        all_code_blocks_combined = '\n'.join(all_code_blocks)
        if all_code_blocks_combined == '':
            return response
        return all_code_blocks_combined
        
    def forward(
        self,
        query,
        expanded_query: str,
        query_type: str,
        file_name: str,
        workspace: str,
    ):
        try_count = 0
        with dspy.settings.context(lm=self.engine):
            if query_type == 'initial':
                result = self.coder(expanded_query=expanded_query, plot_file_name=file_name).code
            else:
                result = self.coder(refine_instruction=expanded_query, plot_file_name=file_name).code
        
        while try_count < 4:
            code = self.get_code(result)
            workspace_structure = print_filesys_struture(workspace)
            
            file_name = f'code_action_{self.model_type}_{query_type}_{try_count}.py'
            with open(os.path.join(workspace, file_name), 'w') as f:
                f.write(code)
            error = None
            log = run_code(workspace, file_name)
            
            if is_run_code_success(log):
                if print_filesys_struture(workspace).find(file_name) == -1:
                    log = log + '\n' + 'No plot generated.'
                    error_message = f'The expected file is not generated. When you complete a plot, remember to save it to a png file. The file name should be """{file_name}"""'
                    
                    # debug and retry
                    try_count += 1
                    with dspy.settings.context(lm=self.engine):
                        result = self.debugger(query=query, err_code=code, error_message=error_message).code
                else:
                    return log, code
            else:
                error = get_error_message(log) if error is None else error
                try_count += 1
                with dspy.settings.context(lm=self.engine):
                    result = self.debugger(query=query, err_code=code, error_message=error).code
        return log, ''
    
    def run(self, query, expanded_query, query_type, file_name, workspace):
        print(f'========Dspy Plot AGENT {query_type} RUN========')
        log, code = self.forward(
            query=query,
            expanded_query=expanded_query,
            query_type=query_type,
            file_name=file_name,
            workspace=workspace,
        )
        return log, code
    