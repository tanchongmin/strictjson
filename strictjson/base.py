import os
import openai
import json
import re
import ast
from openai import OpenAI

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
    - res: Dict. The JSON output of the model. Returns {} if unable to output correct JSON
    '''
    client = OpenAI()
    
    # If OpenAI JSON mode is selected, then just let OpenAI do the processing
    if openai_json_mode:
        # if model fails, default to gpt-3.5-turbo-1106
        try:
            assert(model in ['gpt-4-1106-preview', 'gpt-3.5-turbo-1106'])
        except Exception as e:
            model = 'gpt-3.5-turbo-1106'
            
        output_format_prompt = "\nOutput in the following json format: " + str(output_format) + "\nBe as concise as possible in your output."
            
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

        for i in range(num_tries):

            # make the output format keys with a unique identifier
            new_output_format = {}
            for key in output_format.keys():
                new_output_format[f'{delimiter}{key}{delimiter}'] = output_format[key]
            output_format_prompt = f'''\nOutput in the following json format: {new_output_format}
Output json keys exactly with {delimiter} enclosing keys and perform instructions in the json values.
Be as concise as possible in your output.'''
            
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
            
            # replace the double backslashes meant for content parsing
            res = res.replace('\\n','\n').replace('\\t','\t')

            if verbose:
                print('System prompt:', my_system_prompt)
                print('\nUser prompt:', my_user_prompt)
                print('\nGPT response:', res)

            # try-catch block to ensure output format is adhered to
            try:
                # check key appears for each element in the output
                for key in new_output_format.keys():
                    # if output field missing, raise an error
                    if key not in res: raise Exception(f"{key} not in json output")

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
                    res = end_dict
                    for key in end_dict.keys():
                        try:
                            end_dict[key] = ast.literal_eval(end_dict[key])
                        except Exception as e:
                            # if there is an error in literal processing, do nothing as it is not of the form of a literal
                            continue 

                return end_dict

            except Exception as e:
                error_msg = f"\n\nResult: {res}\n\nError message: {str(e)}\nYou must use \"{delimiter}{{key}}{delimiter}\" to enclose the each {{key}}."
                print("An exception occurred:", str(e))
                print("Current invalid json format:", res)

        return {}
    
# for functions using strict_json
class strict_function:
    def __init__(self, fn_description = 'Output a reminder to define this function in a happy way', 
                 output_format = {'output': 'sentence'}, 
                 examples = None,
                 input_type = None, 
                 output_type = None,
                 **kwargs):
        ''' 
        Creates an LLM-based function using fn_description and outputs JSON based on output_format. 
        Optionally, can define the function based on examples (list of Dict containing input and output variables for each example)
        Optionally, can also force input/output variables to a particular type with input_type and output_type dictionary respectively.
        
        Inputs (compulsory):
        - fn_description: String. Function description to describe process of transforming input variables to output variables
        - output_format: String. Dictionary containing output variables names and description for each variable. There must be at least one output variable
           
        Inputs (optional):
        - examples: Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
        - input_type: Dict. Dictionary containing input variable names as keys and mapping functions as values (need not contain all variables)
        - output_type: Dict. Dictionary containing output variable names as keys and mapping functions as values (need not contain all variables)
        If you do not put any of the optional fields, then we will by default do it by best fit to datatype
        - delimiter: String. The delimiter to enclose input and output keys with
        - kwargs: Dict. Additional arguments you would like to pass on to the strict_json function
        
        ## Example
        fn_description = 'Output the sum of var1 and var2'
        output_format = {'output': 'sum of two numbers'}
        examples = [{'var1': 5, 'var2': 6, 'output': 11}, {'var1': 2, 'var2': 4, 'output': 6}]
        input_type = {'var1': int, 'var2': int}
        output_type = {'output': int}
        
        ## Advanced Conversion of list-based outputs
        - If your output field is of the form of a list, you can ensure strict type conversion of each element using a lambda function
        - Examples
            - For strings, lambda x: [str(y) for y in x]
            - For integers, lambda x: [int(y) for y in x]
        '''
        
        # Compulsary variables
        self.fn_description = fn_description
        self.output_format = output_format
        
        # Optional variables
        self.input_type = input_type
        self.output_type = output_type
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
            kwargs['var'+str(num+1)] = arg
        
        # Do the input type converstion (optional)
        if self.input_type is not None:
            for key in kwargs:
                if key in self.input_type:
                    try:
                        kwargs[key] = self.input_type[key](kwargs[key])
                    except Exception as e: continue

        # do the function. 
        res = strict_json(system_prompt = self.fn_description,
                        user_prompt = kwargs,
                        output_format = self.output_format, 
                        **self.kwargs)
    
        # Do the output type conversion (optional)
        if self.output_type is not None:
            for key in res:
                if key in self.output_type:
                    try:
                        res[key] = self.output_type[key](res[key])      
                    except Exception as e: continue
                
        return res