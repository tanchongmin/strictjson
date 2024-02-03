import os
import openai
import json
import re
import ast
from openai import OpenAI

# to check for datatypes
def check_datatype(field, key, data_type, model):
    ''' Ensures that output field of the key of JSON dictionary is of data_type 
    Currently supports int, float, enum, lists and nested lists'''
    data_type = data_type.strip()
    # check for list at beginning of datatype
    if data_type.lower()[:4] == 'list':
        if not isinstance(field, list):
            # if it is already in a datatype that is a list, ask LLM to fix it (1 LLM call)
            if '[' in field and ']' in field:
                print(f'Attempting to use LLM to fix {field} as it is not a proper list')
                client = OpenAI()
                response = client.chat.completions.create(
                  temperature = 0,
                  model=model,
                  messages=[
                    {"role": "system", "content": '''Output each element of the list in a new line starting with (%item) and ending with \n, e.g. ['hello', 'world'] -> (%item) hello\n(%item) world\nStart your response with (%item) and do not provide explanation'''},
                    {"role": "user", "content": str(field)}
                  ]
                )
                intermediate_response = response.choices[0].message.content

                # Extract out list items
                field = re.findall(r'\(%item\)\s*(.*?)\n*(?=\(%item\)|$)', intermediate_response, flags=re.DOTALL)
                
            else:
                raise Exception(f'''Output field of {key} not of data type list []. If not possible to match, split output field into parts for elements of the list''')
            
    # check for nested list
    # Regex pattern to match content inside square brackets
    match = re.search(r"list\[(.*)\]", data_type, re.IGNORECASE)
    if match:
        internal_data_type = match.group(1)  # Extract the content inside the brackets
        # do processing for internal
        for num in range(len(field)):
            field[num] = check_datatype(field[num], 'element of '+key, internal_data_type, model)
            
    # if it is not nested, check individually
    else:
        # check for string
        if data_type.lower() == 'str':
            try:
                field = str(field)
            except Exception as e:
                pass
            if not isinstance(field, str):
                raise Exception(f"Output field of {key} not of data type {data_type}. If not possible to match, output ''")
        # check for int
        if data_type.lower() == 'int':
            try:
                field = int(field)
            except Exception as e:
                pass
            if not isinstance(field, int):
                raise Exception(f"Output field of {key} not of data type {data_type}. If not possible to match, output 0")
        # check for float
        if data_type.lower() == 'float':
            try:
                field = float(field)
            except Exception as e:
                pass
            if not isinstance(field, float):
                raise Exception(f"Output field of {key} not of data type {data_type}. If not possible to match, output 0.0")
        # check for enum
        if data_type[:4].lower() == 'enum':
            try:
                values = ast.literal_eval(data_type[4:])         
            except:
                raise Exception(f"Enumeration values {data_type[4:]} are not properly defined. Ensure that it is a proper list")
            if field not in values:
                raise Exception(f"Output field of {key} ({field}) not one of {values}. If not possible to match, output {values[0]}")
    return field
                
# for functions using strict_json
def strict_json(system_prompt, user_prompt, output_format, delimiter = '###',
                  model = 'gpt-3.5-turbo', temperature = 0, num_tries = 3, verbose = False, literal_eval = True, openai_json_mode = False, **kwargs):
    ''' Ensures that OpenAI will always adhere to the desired output JSON format defined in output_format. 
    Uses rule-based iterative feedback to ask GPT to self-correct.
    Keeps trying up to num_tries it it does not. Returns empty JSON if unable to after num_tries iterations.
    
    Inputs:
    - system_prompt: String. Write in whatever you want GPT to become. e.g. "You are a \<purpose in life\>"
    - user_prompt: String. The user input. Later, when we use it as a function, this is the function input
    - output_format: JSON. JSON format with the key as the output key, and the value as the output description
    - delimiter: String. This is the delimiter to surround the keys. With delimiter ###, key becomes ###key###
    - model: String. The OpenAI model to use for json generation
    - temperature: Float (default: 0) The temperature of the openai model, the higher the more variable the output (lowest = 0)
    - num_tries: Integer (default: 3) The number of tries to iteratively prompt GPT to generate correct json format
    - verbose: Boolean (default: False). Whether or not to print out the system prompt, user prompt, GPT response
    - literal_eval: Boolean (default: True). Whether or not to do ast.literal_eval for output fields
    - openai_json_mode: Boolean (default: False). Whether or not to use OpenAI JSON Mode
    - **kwargs: Dict. Additional arguments you would like to pass on to OpenAI Chat Completion API
    
    Output:
    - res: Dict. The JSON output of the model. Returns {} if JSON parsing failed.
    '''
    client = OpenAI()
    
    # If OpenAI JSON mode is selected, then just let OpenAI do the processing
    if openai_json_mode:
        # if model fails, default to gpt-3.5-turbo-1106
        try:
            assert(model in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106'])
        except Exception as e:
            model = 'gpt-3.5-turbo-1106'
            
        output_format_prompt = "\nOutput in the following json string format: " + str(output_format) + "\nBe concise."
            
        my_system_prompt = str(system_prompt) + output_format_prompt
        my_user_prompt = str(user_prompt) 
            
        response = client.chat.completions.create(
            temperature = temperature,
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": my_system_prompt},
                {"role": "user", "content": my_user_prompt}
            ],
            **kwargs
        )
        res = response.choices[0].message.content
        
        if verbose:
            print('System prompt:', my_system_prompt)
            print('\nUser prompt:', my_user_prompt)
            print('\nGPT response:', res)
        try:
            loaded_json = json.loads(res)
        except Exception as e:
            loaded_json = {}
        return loaded_json
        
    # Otherwise, implement JSON parsing using Strict JSON
    else:
        # start off with no error message
        error_msg = ''
        
        # make the output format keys with a unique identifier
        new_output_format = {}
        for key in output_format.keys():
            new_output_format[f'{delimiter}{key}{delimiter}'] = '<'+str(output_format[key])+'>'
        output_format_prompt = f'''\nOutput in the following json string format: {new_output_format}
You must update text within <>. Be concise.'''

        for i in range(num_tries):
            my_system_prompt = str(system_prompt) + output_format_prompt + error_msg
            my_user_prompt = str(user_prompt) 

            # Use OpenAI to get a response
            client = OpenAI()
            response = client.chat.completions.create(
              temperature = temperature,
              model=model,
              messages=[
                {"role": "system", "content": my_system_prompt},
                {"role": "user", "content": my_user_prompt}
              ],
              **kwargs
            )

            res = response.choices[0].message.content
            
            # remove all escape characters so that code can be run
            res = bytes(res, "utf-8").decode("unicode_escape")

            if verbose:
                print('System prompt:', my_system_prompt)
                print('\nUser prompt:', my_user_prompt)
                print('\nGPT response:', res)

            # try-catch block to ensure output format is adhered to
            try:
                # check key appears for each element in the output
                for key in new_output_format.keys():
                    # if output field missing, raise an error
                    if key not in res: 
                        # try to fix it if possible
                        if res.count(f"'{key}':") == 1:
                            res = res.replace(f"'{key}':", f"'{delimiter}{key}{delimiter}':")
                        elif res.count(f'"{key}"') == 1:
                            res = res.replace(f'"{key}":', f'"{delimiter}{key}{delimiter}":')
                        else:
                            raise Exception(f"{key} not in json output. You must use \"{delimiter}{{key}}{delimiter}\" to enclose the {{key}}.")

                # if all is good, we then extract out the fields
                # Use regular expressions to extract keys and values
                pattern = fr",*\s*['|\"]{delimiter}([^#]*){delimiter}['|\"]: "

                matches = re.split(pattern, res[1:-1])

                # remove null matches
                my_matches = [match for match in matches if match !='']

                # remove the ' from the value matches
                curated_matches = [match[1:-1] if match[0] in '\'"' else match for match in my_matches]

                # create a dictionary
                end_dict = {}
                for i in range(0, len(curated_matches), 2):
                    end_dict[curated_matches[i]] = curated_matches[i+1]

                # try to do some parsing via literal_eval
                if literal_eval:
                    for key in end_dict.keys():
                        try:
                            end_dict[key] = ast.literal_eval(end_dict[key])
                        except Exception as e:
                            # if there is an error in literal processing, do nothing as it is not of the form of a literal
                            continue 
                   
                # check for data types
                for key in end_dict.keys():
                    # check for data type if any
                    if key in output_format and 'type:' in output_format[key]:
                        # extract out data type
                        data_type = output_format[key].split('type:')[-1]
                        # check the data type, perform type conversion as necessary
                        end_dict[key] = check_datatype(end_dict[key], key, data_type, model)       

                return end_dict

            except Exception as e:
                error_msg = f"\n\nPrevious json: {res}\njson error: {str(e)}\nFix the error."
                print("An exception occurred:", str(e))
                print("Current invalid json format:", res)

        return {}
    
# for functions using strict_json
class strict_function:
    def __init__(self, fn_description = 'Output a reminder to define this function in a happy way', 
                 output_format = {'output': 'sentence'}, 
                 variable_names = [],
                 examples = None,
                 **kwargs):
        ''' 
        Creates an LLM-based function using fn_description and outputs JSON based on output_format. 
        Optionally, can define the function based on examples (list of Dict containing input and output variables for each example)
        Optionally, can also force input/output variables to a particular type with input_type and output_type dictionary respectively.
        
        Inputs (compulsory):
        - fn_description: String. Function description to describe process of transforming input variables to output variables
        - output_format: String. Dictionary containing output variables names and description for each variable. There must be at least one output variable
           
        Inputs (optional):
        - **variable_names** - How the variables should be named in a list
        - examples: Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
        - delimiter: String. The delimiter to enclose input and output keys with
        - kwargs: Dict. Additional arguments you would like to pass on to the strict_json function
        
        ## Example
        fn_description = 'Output the sum of num1 and num2'
        output_format = {'output': 'sum of two numbers'}
        variable_names = ['num1', 'num2']
        examples = [{'num1': 5, 'num2': 6, 'output': 11}, {'num1': 2, 'num2': 4, 'output': 6}]
        '''
        
        # Compulsary variables
        self.fn_description = fn_description
        self.output_format = output_format
        
        # Optional variables
        self.variable_names = variable_names
        self.examples = examples
        self.kwargs = kwargs
        
        if self.examples is not None:
            self.fn_description += '\nExamples:\n' + str(examples)
        
    def __call__(self, *args, **kwargs):
        ''' Describes the function, and inputs the relevant parameters as either unnamed variables (args) or named variables (kwargs)
        If there is any variable that needs to be strictly converted to a datatype, put mapping function in input_type or output_type
        
        Inputs:
        - *args: Tuple. Unnamed input variables of the function. Will be processed to var1, var2 and so on based on order in the tuple
        - **kwargs: Dict. Named input variables of the function
        
        Output:
        - res: Dict. JSON containing the output variables'''
        
        # Do the merging of args and kwargs
        for num, arg in enumerate(args):
            if len(self.variable_names) > num:
                kwargs[self.variable_names[num]] = arg
            else:
                kwargs['var'+str(num+1)] = arg

        # do the function. 
        res = strict_json(system_prompt = self.fn_description,
                        user_prompt = kwargs,
                        output_format = self.output_format, 
                        **self.kwargs)
                
        return res