import os
import openai
import json
import re
import ast
import copy
from openai import OpenAI

### Helper Functions ###

def convert_to_list(field: str, **kwargs) -> list:
    '''Converts the string field into a list using the LLM (with **kwargs) to list out elements line by line'''
    
    system_msg = '''Output each element of the list in a new line starting with (%item) and ending with \n, e.g. ['hello', 'world'] -> (%item) hello\n(%item) world\nStart your response with (%item) and do not provide explanation'''
    user_msg = str(field)
    res = chat(system_msg, user_msg, **kwargs)

    # Extract out list items
    field = re.findall(r'\(%item\)\s*(.*?)\n*(?=\(%item\)|$)', res, flags=re.DOTALL)
    return field

def convert_to_dict(field: str, keys: dict, delimiter: str) -> dict:
    '''Converts the string field into a dictionary with keys by splitting on '{delimiter}{key}{delimiter}' '''
    output_d = {}
    for key in keys:
        # if output field missing, raise an error
        if f"'{delimiter}{key}{delimiter}':" not in field and f'"{delimiter}{key}{delimiter}":' not in field: 
            # try to fix it if possible
            if field.count(f"'{key}':") == 1:
                field = field.replace(f"'{key}':", f"'{delimiter}{key}{delimiter}':")
            elif field.count(f'"{key}"') == 1:
                field = field.replace(f'"{key}":', f'"{delimiter}{key}{delimiter}":')
            else:
                raise Exception(f'''"{key}" not in json string output. You must use \"{delimiter}{{key}}{delimiter}\" to enclose the {{key}}.''')

    # if all is good, we then extract out the fields
    # Use regular expressions to extract keys and values
    pattern = fr",*\s*['|\"]{delimiter}([^#]*){delimiter}['|\"]: "

    matches = re.split(pattern, field[1:-1])

    # remove null matches
    my_matches = [match for match in matches if match !='']

    # remove the ' from the value matches
    curated_matches = [match[1:-1] if match[0] in '\'"' else match for match in my_matches]

    # create a dictionary
    for i in range(0, len(curated_matches), 2):
        output_d[curated_matches[i]] = curated_matches[i+1]
        
    return output_d

def llm_check(field, llm_check_msg: str, **kwargs) -> (bool, str):
    ''' Uses the LLM to check if the field adheres to the llm_check_msg.
    Outputs whether requirement is met (True or False) and the action needed'''
    system_msg = f'''Check whether output field meets this requirement: {llm_check_msg}
Output in the following format:
```
# Thoughts: <Thoughts about whether output field meets requirement>
# Requirement Met: <Yes or No>
# Action Needed: <If Requirement Met is No, state in one sentence how to meet requirement. Otherwise, output NA>"
```
Update text enclosed in <>. Be concise.
'''
    user_msg = str(field)
    res = chat(system_msg, user_msg, **kwargs)
    
    pattern = r"# Thoughts: (.+)\n# Requirement Met: (.+)\n# Action Needed: (.+)"
    matches = re.findall(pattern, res)

    if matches:
        thoughts, requirement_met, action_needed = matches[0]
        if 'yes' in requirement_met.lower():
            requirement_met = True
        else:
            requirement_met = False
        # print("Thoughts:", thoughts)
        print("Requirement Met:", requirement_met)
        if not requirement_met:
            print("Action Needed:", action_needed)
        print('\n')
    else:
        # if regex failed to parse, we just check for yes / no. And append whole string as action needed
        if 'yes' in res.lower():
            thoughts, requirement_met, action_needed = '', True, ''
        else:
            thoughts, requirement_met, action_needed = '', False, res
            
    return requirement_met, action_needed

def check_datatype(field, key: dict, data_type: str, **kwargs):
    ''' Ensures that output field of the key of JSON dictionary is of data_type 
    Currently supports int, float, enum, lists and nested lists
    Takes in **kwargs for the LLM model
    Returns corrected output field that matches the datatype'''
    data_type = data_type.strip()
    
    # check if we want an LLM-based correction
    if data_type.lower()[:6] == 'ensure':
        llm_check_msg = data_type[6:].strip()
        print(f'Using LLM to check "{field}" to see if it adheres to "{llm_check_msg}"')
        requirement_met, action_needed = llm_check(field, llm_check_msg, **kwargs)
        # if check failed, raise error
        if not requirement_met:
            raise Exception(f'''Output field of "{key}" does not meet requirement "{llm_check_msg}". Action needed: "{action_needed}"''')
            
    # check for list at beginning of datatype
    # or the output field begins with [ and ends with ] but it is not a list, indicating an error with ast.literal_eval
    if data_type.lower()[:4] == 'list' or (str(field)[0]=='[' and str(field)[-1]==']'):
        if not isinstance(field, list):
            # if it is already in a datatype that is a list, ask LLM to fix it (1 LLM call)
            if '[' in field and ']' in field:
                print(f'Attempting to use LLM to fix {field} as it is not a proper list')
                field = convert_to_list(field, **kwargs)   
                print(f'Fixed list: {field}\n\n')
            else:
                raise Exception(f'''Output field of "{key}" not of data type list []. If not possible to match, split output field into parts for elements of the list''')
            
    # check for nested list
    # Regex pattern to match content inside square brackets
    match = re.search(r"list\[(.*)\]", data_type, re.IGNORECASE)
    if match:
        internal_data_type = match.group(1)  # Extract the content inside the brackets
        # do processing for internal elements
        for num in range(len(field)):
            field[num] = check_datatype(field[num], 'list element of '+key, internal_data_type, **kwargs)
            
    # if it is not nested, check individually
    else:
        # check for string
        if data_type.lower() == 'str':
            try:
                field = str(field)
            except Exception as e:
                pass
            if not isinstance(field, str):
                raise Exception(f'''Output field of "{key}" not of data type {data_type}. If not possible to match, output '' ''')
                
        # check for int
        if data_type.lower() == 'int':
            try:
                field = int(field)
            except Exception as e:
                pass
            if not isinstance(field, int):
                raise Exception(f'Output field of "{key}" not of data type {data_type}. If not possible to match, output 0')
        
        # check for float
        if data_type.lower() == 'float':
            try:
                field = float(field)
            except Exception as e:
                pass
            if not isinstance(field, float):
                raise Exception(f'Output field of "{key}" not of data type {data_type}. If not possible to match, output 0.0')
                
        # check for bool
        if data_type.lower() == 'bool':
            field = str(field)
            if field[:4].lower() == 'true':
                field = True
            elif field[:5].lower() == 'false':
                field = False
            else:
                raise Exception(f'Output field of "{key}" not of data type {data_type}. If not possible to match, output True')

        # check for dict
        if data_type[:4].lower() == 'dict':
            if not isinstance(field, dict):
                raise Exception(f"Output field of {key} not of data type dict. If not possible to match, rephrase output field into dictionary with attribute names as key and attribute description as value")
            # if we define more things in dict, evaluate those
            if len(data_type) > 4:
                try:
                    attribute_checks = ast.literal_eval(data_type[4:])
                    assert(isinstance(attribute_checks, list) == True)
                except Exception as e:
                    raise Exception(f'Dictionary keys {data_type[4:]} of output field of "{key}" are not properly defined. Ensure that it is a proper list')
                    
                # if data_type is a valid list, check if elements of list are present in dictionary
                if isinstance(attribute_checks, list):
                    for item in attribute_checks:
                        if item not in field.keys():
                            raise Exception(f'Output field of "{key}" of type dict does not contain the key "{item}". The dict should contain keys {attribute_checks}')
                
        # check for enum
        if data_type[:4].lower() == 'enum':
            try:
                values = ast.literal_eval(data_type[4:])  
                assert(isinstance(values, list) == True)
            except Exception as e:
                raise Exception(f'Enumeration values {data_type[4:]} of output field of "{key}" are not properly defined. Ensure that it is a proper list')
            if field not in values:
                raise Exception(f'Output field of "{key}" ({field}) not one of {values}. If not possible to match, output {values[0]}')
    return field

def check_key(field: str, output_format, new_output_format, delimiter: str, delimiter_num: int, **kwargs):
    ''' Check whether each key in dict, or elements in list of new_output_format is present in field, and whether they meet the right data type requirements, then convert field to the right data type
    If needed, calls LLM model with parameters **kwargs to correct the output format for improperly formatted list
    output_format is user-given output format at each level, new_output_format is with delimiters in keys, and angle brackets surrounding values
    If output_format is a string, decode escape characters, so that code can run
    Returns field that is converted to a dictionary if able to. Otherwise, raises Exception errors for missing keys or wrong output format'''
    
    cur_delimiter = delimiter*delimiter_num
    
    if isinstance(output_format, dict):   
        # this is the processed output dictionary for that particular layer in the output structure
        output_d = {}
        # check key appears for each element in the output
        output_d = convert_to_dict(field, output_format.keys(), cur_delimiter)
            
        # after creating dictionary, step into next layer
        for key, value in output_d.items():
            output_d[key] = check_key(value, output_format[key], new_output_format[cur_delimiter+key+cur_delimiter], delimiter, delimiter_num+1)
            # after stepping back from the later layers back to present layer, check for types
            if 'type:' in output_format[key]:             
                # extract out data type
                data_type = output_format[key].split('type:')[-1]
                # check the data type, perform type conversion as necessary
                output_d[key] = check_datatype(output_d[key], key, data_type, **kwargs)   
                
        return output_d

    # if list, step into each element
    elif isinstance(output_format, list):
        try:
            field = ast.literal_eval(field)
        except Exception as e:
            # if there is an error in literal processing, use LLM to split field into list
            field = convert_to_list(field, **kwargs)
            
        # check that list has at least same number of elements as the input
        if len(field) < len(output_format):
            raise Exception(f'''Output "{field}" has fewer elements than required by "{output_format}". Add in more list elements.''')
        
        return [check_key(str(field[num]), output_format[num], new_output_format[num], delimiter, delimiter_num+1) for num in range(len(output_format))]
    
    # if string, then do literal eval, then decode unicode escape characters for code to run
    elif isinstance(output_format, str):
        # if literal eval fails, just leave it as string, no need to raise error
        try:
            field = ast.literal_eval(field)
        except Exception as e:
            pass
        return remove_unicode_escape(field)
    
    # otherwise just return the value
    else:
        return field
    
def remove_unicode_escape(my_datatype):
    ''' Removes the unicode escape character from the ending string in my_datatype (can be nested) '''
    if isinstance(my_datatype, dict):
        output_d = {}
        # wrap keys with delimiters
        for key, value in my_datatype.items():
            output_d[key] = remove_unicode_escape(value)
        return output_d
    elif isinstance(my_datatype, list):
        return [remove_unicode_escape(item) for item in my_datatype]
    # if it is a string, remove the unicode escape characters from it, so code can be run
    elif isinstance(my_datatype, str):
        # only do decoding for code if backslash present
        if '\\' in my_datatype:
            my_datatype = bytes(my_datatype, "utf-8").decode("unicode_escape")
        # replace aprostrophes
        my_datatype = my_datatype.replace("â\x80\x99", "'")
        return my_datatype
    else:
        return my_datatype
    
def wrap_with_angle_brackets(d: dict, delimiter: str, delimiter_num: int) -> dict:
    ''' Changes d to output_d by wrapping delimiters over the keys, and putting angle brackets on the values '''
    if isinstance(d, dict):
        output_d = {}
        # wrap keys with delimiters
        for key, value in d.items():
            new_key = f'{delimiter}'*delimiter_num + str(key) + f'{delimiter}'*delimiter_num
            output_d[new_key] = wrap_with_angle_brackets(value, delimiter, delimiter_num+1)
        return output_d
    elif isinstance(d, list):
        return [wrap_with_angle_brackets(item, delimiter, delimiter_num+1) for item in d]
    elif isinstance(d, str):
        return f'<{d}>'
    else:
        return d
    
def chat(system_prompt: str, user_prompt: str, model: str = 'gpt-3.5-turbo', temperature: float = 0, verbose: bool = False, host: str = 'openai', llm = None, **kwargs):
    '''Performs a chat with the host's LLM model with system prompt, user prompt, model, verbose and kwargs
    Returns the output string res
    - system_prompt: String. Write in whatever you want the LLM to become. e.g. "You are a \<purpose in life\>"
    - user_prompt: String. The user input. Later, when we use it as a function, this is the function input
    - model: String. The LLM model to use for json generation
    - verbose: Boolean (default: False). Whether or not to print out the system prompt, user prompt, GPT response
    - host: String. The provider of the LLM
    - llm: User-made llm function.
        - Inputs:
            - system_prompt: String. Write in whatever you want the LLM to become. e.g. "You are a \<purpose in life\>"
            - user_prompt: String. The user input. Later, when we use it as a function, this is the function input
        - Output:
            - res: String. The response of the LLM call
    - **kwargs: Dict. Additional arguments for LLM chat
    
    TODO: Incorporate other open-sourced LLMs in the future'''
    if llm is not None:
        ''' If you specified your own LLM, then we just feed in the system and user prompt 
        LLM function should take in system prompt (str) and user prompt (str), and output a response (str) '''
        res = llm(system_prompt = system_prompt, user_prompt = user_prompt)
    
    ## This part here is for llms that are OpenAI based
    elif host == 'openai':
        # additional checks for openai json mode
        if 'response_format' in kwargs and kwargs['response_format'] == {"type": "json_object"}:
            # if model fails, default to gpt-3.5-turbo-1106
            try:
                assert(model in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106'])
            except Exception as e:
                model = 'gpt-3.5-turbo-1106'
                
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            temperature = temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )
        res = response.choices[0].message.content

    if verbose:
        print('System prompt:', system_prompt)
        print('\nUser prompt:', user_prompt)
        print('\nGPT response:', res)
            
    return res

### Main Functions ###
                
def strict_json(system_prompt: str, user_prompt: str, output_format: dict, custom_checks: dict = {}, check_data = None, delimiter: str = '###', num_tries: int = 3, openai_json_mode: bool = False, **kwargs):
    ''' Ensures that OpenAI will always adhere to the desired output JSON format defined in output_format. 
    Uses rule-based iterative feedback to ask GPT to self-correct.
    Keeps trying up to num_tries it it does not. Returns empty JSON if unable to after num_tries iterations.
    
    Inputs (compulsory):
    - system_prompt: String. Write in whatever you want GPT to become. e.g. "You are a \<purpose in life\>"
    - user_prompt: String. The user input. Later, when we use it as a function, this is the function input
    - output_format: Dict. JSON format with the key as the output key, and the value as the output description
    
    Inputs (optional):
    - custom_checks: Dict. Key is output key, value is function which does checking of content for output field
    - check_data: Any data type. The additional data for custom_checks to use if required
    - delimiter: String (Default: '###'). This is the delimiter to surround the keys. With delimiter ###, key becomes ###key###
    - num_tries: Integer (default: 3). The number of tries to iteratively prompt GPT to generate correct json format
    - openai_json_mode: Boolean (default: False). Whether or not to use OpenAI JSON Mode
    - **kwargs: Dict. Additional arguments for LLM chat
    
    Output:
    - res: Dict. The JSON output of the model. Returns {} if JSON parsing failed.
    '''
    # If OpenAI JSON mode is selected, then just let OpenAI do the processing
    if openai_json_mode:
        output_format_prompt = "\nOutput in the following json string format: " + str(output_format) + "\nBe concise."
            
        my_system_prompt = str(system_prompt) + output_format_prompt
        my_user_prompt = str(user_prompt) 
            
        res = chat(my_system_prompt, my_user_prompt, response_format = {"type": "json_object"}, **kwargs)
            
        try:
            loaded_json = json.loads(res)
        except Exception as e:
            loaded_json = {}
        return loaded_json
        
    # Otherwise, implement JSON parsing using Strict JSON
    else:
        # start off with no error message
        error_msg = ''

        # wrap the values with angle brackets and wrap keys with delimiter to encourage LLM to modify it
        new_output_format = wrap_with_angle_brackets(output_format, delimiter, 1)
        
        output_format_prompt = f'''\nOutput in the following json string format: {new_output_format}
Update text enclosed in <>. Be concise. Output only the json string without any explanation.'''

        for i in range(num_tries):
            my_system_prompt = str(system_prompt) + output_format_prompt + error_msg
            my_user_prompt = str(user_prompt) 

            # Use OpenAI to get a response
            res = chat(my_system_prompt, my_user_prompt, **kwargs)

            # try-catch block to ensure output format is adhered to
            try:
                # check that res is a json string
                if res[0] != '{' or res[-1] != '}':
                    raise Exception('Ensure output must be a json string beginning with { and ending with }')
                
                # do checks for keys and output format, remove escape characters so code can be run
                end_dict = check_key(res, output_format, new_output_format, delimiter, delimiter_num = 1, **kwargs)
                
                # run user defined custom checks now
                for key in end_dict:
                    if key in custom_checks:
                        for check in custom_checks[key]:
                            requirement, requirement_met, action_needed = check(end_dict[key], check_data)
                            print(f'Running check for "{requirement}" on output field of "{key}"')
                            if not requirement_met:
                                print(f'Requirement not met. Action needed: "{action_needed}"\n\n')
                                raise Exception(f'Output field of "{key}" does not meet requirement "{requirement}". Action needed: "{action_needed}"')
                            else:
                                print('Requirement met\n\n')
                return end_dict

            except Exception as e:
                error_msg = f"\n\nPrevious json: {res}\njson error: {str(e)}\nFix the error."                
                print("An exception occurred:", str(e))
                print("Current invalid json format:", res)

        return {}

class Function:
    def __init__(self,
                 fn_description: str = 'Output a reminder to define this function in a happy way', 
                 output_format: dict = {'output': 'sentence'},
                 examples = None,
                 external_fn = None,
                 fn_name = None,
                 **kwargs):
        ''' 
        Creates an LLM-based function or wraps an external function using fn_description and outputs JSON based on output_format. 
        (Optional) Can define the function based on examples (list of Dict containing input and output variables for each example)
        (Optional) If you would like greater specificity in your function's input, you can describe the variable after the : in the input variable name, e.g. `<var1: an integer from 10 to 30`. Here, `var1` is the input variable and `an integer from 10 to 30` is the description.
        
        Inputs (compulsory):
        - fn_description: String. Function description to describe process of transforming input variables to output variables. Variables must be enclosed in <> and listed in order of appearance in function input.
        - output_format: String. Dictionary containing output variables names and description for each variable.
           
        Inputs (optional):
        - examples: Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
        - external_fn: Python Function. If defined, instead of using LLM to process the function, we will run the external function. 
            If there are multiple outputs of this function, we will map it to the keys of `output_format` in a one-to-one fashion
        - fn_name: String. If provided, this will be the name of the function. Otherwise, if `external_fn` is provided, it will be the name of `external_fn`. Otherwise, we will use LLM to generate a function name from the `fn_description`
        - **kwargs: Dict. Additional arguments you would like to pass on to the strict_json function
        
        ## Example
        fn_description = 'Output the sum of <num1> and <num2>'
        output_format = {'output': 'sum of two numbers'}
        examples = [{'num1': 5, 'num2': 6, 'output': 11}, {'num1': 2, 'num2': 4, 'output': 6}]
        '''
        
        # Compulsary variables
        self.fn_description = fn_description
        self.output_format = output_format
        
        # Optional variables
        self.examples = examples
        self.external_fn = external_fn
        self.fn_name = fn_name
        self.kwargs = kwargs
        
        self.variable_names = []
        # use regex to extract variables from function description
        matches = re.findall(r'<(.*?)>', self.fn_description)

        for match in matches:
            first_half = match.split(':')[0]
            if first_half not in self.variable_names:
                self.variable_names.append(first_half)
                
        # generate function name if not defined
        if self.fn_name is None:
            # if external function has a name, use it
            if self.external_fn is not None and hasattr(self.external_fn, '__name__'):
                self.fn_name = self.external_fn.__name__
            # otherwise, generate name out
            else:
                res = strict_json(system_prompt = "Output a function name to summarise the usage of this function.",
                                  user_prompt = str(self.fn_description),
                                  output_format = {"Thoughts": "What function does", "Name": "Function name with _ separating words that summarises what function does"})
                self.fn_name = res['Name']
                
        # change instance's name to function's name
        self.__name__ = self.fn_name

        # Append examples to description
        if self.examples is not None:
            self.fn_description += '\nExamples:\n' + str(examples)
      
    def __str__(self):
        ''' Prints out the function's parameters '''
        return f'Description: {self.fn_description}\nInput: {self.variable_names}\nOutput: {self.output_format}\n'
        
    def __call__(self, *args, **kwargs):
        ''' Describes the function, and inputs the relevant parameters as either unnamed variables (args) or named variables (kwargs)
        
        Inputs:
        - *args: Tuple. Unnamed input variables of the function. Will be processed to var1, var2 and so on based on order in the tuple
        - **kwargs: Dict. Named input variables of the function. Can also be variables to pass into strict_json
        
        Output:
        - res: Dict. JSON containing the output variables'''
        
        # extract out only variables listed in variable_list
        function_kwargs = {my_key: kwargs[my_key] for my_key in kwargs if my_key in self.variable_names}
        # extract out only variables not listed in variable list
        strict_json_kwargs = {my_key: kwargs[my_key] for my_key in kwargs if my_key not in self.variable_names}
        
        # Do the auto-naming of variables as var1, var2, or as variable names defined in variable_names
        for num, arg in enumerate(args):
            if len(self.variable_names) > num:
                function_kwargs[self.variable_names[num]] = arg
            else:
                function_kwargs['var'+str(num+1)] = arg
                
        # If strict_json function, do the function. 
        if self.external_fn is None:
            res = strict_json(system_prompt = self.fn_description,
                            user_prompt = function_kwargs,
                            output_format = self.output_format, 
                            **self.kwargs, **strict_json_kwargs)
            
        # Else run the external function
        else:
            res = {}
            fn_output = self.external_fn(**function_kwargs)
            output_keys = list(self.output_format.keys())
            # convert the external function into a tuple format to parse it through the JSON dictionary output format
            if not isinstance(fn_output, tuple):
                fn_output = [fn_output]
            
            for i in range(len(fn_output)):
                res[output_keys[i]] = fn_output[i]
                
        return res
    
### Legacy Support ###
# alternative names for strict_json
strict_text = strict_json
strict_output = strict_json

# alternative name for strict_function (it is now called Function)
strict_function = Function