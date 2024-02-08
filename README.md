# Strict JSON
### A Strict JSON Framework for LLM Outputs, that fixes problems that json.loads() cannot solve
- Works for JSON outputs with multiple ' or " or { or } or \ or unmatched braces/brackets that may break a json.loads()

### Functionalities (v2.2.1) 8 Feb 2024:
- Ensures LLM outputs into a dictionary based on a JSON format (HUGE: Nested lists and dictionaries now supported)
- Supports `int`, `float`, `str`, `dict`, `list`, `Dict[]`, `List[]`, `Enum[]` type forcing with LLM-based error correction, as well as LLM-based error correction using `type: ensure <restriction>`, and (advanced) custom user checks using `custom_checks`
- Easy construction of LLM-based functions using ```strict_function```
- Easy integration with OpenAI JSON Mode by setting `openai_json_mode = True`

### Tutorials and Community Support
- Created: 28 Oct 2023
- Collaborators welcome
- Video tutorial: [https://www.youtube.com/watch?v=IjTUKAciTCg](https://www.youtube.com/watch?v=1N-znDTlhNc)
- Discussion Channel (my discord - John's AI Group): [discord.gg/bzp87AHJy5](discord.gg/bzp87AHJy5)


## How do I use this? 
1. Download package via command line ```pip install strictjson```
2. Set up your OpenAPI API Key. Refer to ```Tutorial.ipynb``` for how to do it for Jupyter Notebooks.
3. Import the required functions from ```strictjson``` and use them!

## How does it work?
- Extract JSON values as a string using a special regex (add delimiters to ```key``` to make ```###key###```) to split keys and values. (New!) Also works for nested datatypes by splitting recursively.
- Uses ```ast.literal_eval``` to best match the extracted output value to a literal (e.g. int, string, dict).
- Ensures that all JSON fields are output by LLM, with optional type checking, if not it will feed in error message to LLM to iteratively correct its generation (default: 3 tries)

# Features:
## 1. Basic Generation
- **system_prompt**: Write in whatever you want the LLM to become. "You are a \<purpose in life\>"
- **user_prompt**: The user input. Later, when we use it as a function, this is the function input
- **output_format**: JSON of output variables in a dictionary, with the key as the output key, and the value as the output description
    - The output keys will be preserved exactly, while GPT will generate content to match the description of the value as best as possible
 
#### Example Usage
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment',
                                    'Adjectives': 'List of adjectives',
                                    'Words': 'Number of words'})
                                    
print(res)
```

#### Example output
```{'Sentiment': 'positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7}```

## 2. Advanced Generation
- More advanced demonstration involving code that would typically break ```json.loads()```

#### Example Usage
```python
res = strict_json(system_prompt = 'You are a code generator, generating code to fulfil a task',
                    user_prompt = 'Given array p, output a function named func_sum to return its sum',
                    output_format = {'Elaboration': 'How you would do it',
                                     'C': 'Code',
                                    'Python': 'Code'})
                                    
print(res)
```

#### Example output
```{'Elaboration': 'To calculate the sum of an array, we can iterate through each element of the array and add it to a running total.', ```

```'C': 'int func_sum(int p[], int size) {\n    int sum = 0;\n    for (int i = 0; i < size; i++) {\n        sum += p[i];\n    }\n    return sum;\n}', ```

```'Python': 'def func_sum(p):\n    sum = 0\n    for num in p:\n        sum += num\n    return sum'}```

## 3. Type forcing
- Generally, ```strict_json``` will infer the data type automatically for you for the output fields
- However, if you would like very specific data types, you can do data forcing using ```type: <data_type>``` at the last part of the output field description
- ```<data_type>``` must be of the form `int`, `float`, `str`, `dict`, `list`, `Dict[]`, `List[]`, `Enum[]`
- The `Enum` and `List` are not case sensitive, so `enum` and `list` works just as well
- For `Enum[list_of_category_names]`, it is best to give an "Other" category in case the LLM fails to classify correctly with the other options.
- If `list` or `List[]` is not formatted correctly in LLM's output, we will correct it by asking the LLM to list out the elements line by line
- For `dict`,  we can further check whether keys are present using `Dict[list_of_key_names]`
- Other types will first be forced by rule-based conversion, any further errors will be fed into LLM's error feedback mechanism
- If `<data_type>` is not the specified data types, it can still be useful to shape the output for the LLM. However, no type checking will be done.

### LLM-based checks
- If you would like the LLM to ensure that the type is being met, use `type: ensure <requirement>`
- This will run a LLM to check if the requirement is met. If requirement is not met, the LLM will generate what needs to be done to meet the requirement, which will be fed into the error-correcting loop of `strict_json`

#### Example Usage 1
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment, type: Enum["Pos", "Neg", "Other"]',
                                    'Adjectives': 'List of adjectives, type: List[str]',
                                    'Words': 'Number of words, type: int'})
                                    
print(res)
```

#### Example output 1
```{'Sentiment': 'Pos', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7}```

#### Example Usage 2
```python
res = strict_json(system_prompt = 'You are an expert at organising birthday parties',
                    user_prompt = 'Give me some information on how to organise a birthday',
                    output_format = {'Famous Quote': 'quote with name, type: ensure quote contains the word age',
                                    'Lucky draw numbers': '3 numbers from 1-50, type: List[int]',
                                    'Sample venues': 'Describe two venues, type: List[Dict["Venue", "Description"]]'})

print(res)
```

#### Example output 2
```{'Famous Quote': 'The secret of staying young is to live honestly, eat slowly, and lie about your age. - Lucille Ball',```
```'Lucky draw numbers': [7, 21, 35],```

```'Sample venues': [{'Venue': 'Beachside Resort', 'Description': 'A beautiful resort with stunning views of the beach. Perfect for a summer birthday party.'}, {'Venue': 'Indoor Trampoline Park', 'Description': 'An exciting venue with trampolines and fun activities. Ideal for an active and energetic birthday celebration.'}]}```

## 4. Strict JSON Functions
- Enhances ```strict_json()``` with a function-like interface for repeated use of modular LLM-based functions
- Use angle brackets <> to enclose variable names
- Inputs (compulsory):
    - **fn_description** - Function description to describe process of transforming input variables to output variables
    - **output_format** - Dictionary containing output variables names and description for each variable. There must be at least one output variable
- Inputs (optional):
    - **variable_names** - How the variables should be named in a list, if you don't want to use the default var1, var2, e.g.
    - **examples** - Examples in Dictionary form with the input and output variables (list if more than one)
    - **kwargs** - Additional arguments you would like to pass on to the ```strict_json``` function
        
- Outputs:
    JSON of output variables in a dictionary (similar to ```strict_json```)
    
#### Example Usage 1 (Description only)
```python
# Construct the function: var1 will be first input variable, var2 will be second input variable and so on
fn = strict_function(fn_description = 'Output a sentence with words <var1> and <var2> in the style of <var3>', 
                     output_format = {'output': 'sentence'})

# Use the function
fn('ball', 'dog', 'happy')
```

#### Example Output 1
```{'output': 'The happy dog chased the ball.'}```

#### Example Usage 2 (Examples only)
```python
# Construct the function: infer pattern from just examples without description (here it is multiplication)
fn = strict_function(fn_description = 'Map input to output based on examples', 
                     output_format = {'output': 'final answer'}, 
                     examples = [{'var1': 3, 'var2': 2, 'output': 6}, 
                                 {'var1': 5, 'var2': 3, 'output': 15}, 
                                 {'var1': 7, 'var2': 4, 'output': 28}])

# Use the function
fn(2, 10)
```

#### Example Output 2
```{'output': 20}```

#### Example Usage 3 (Description and Variable Names and Examples)
```python
# Construct the function: description and examples with variable names
# variable names will be referenced in order of input
fn = strict_function(fn_description = 'Output the sum and difference of <num1> and <num2>', 
                 output_format = {'sum': 'sum of two numbers', 
                                  'difference': 'absolute difference of two numbers'}, 
                 variable_names = ['num1', 'num2'],
                 examples = {'num1': 2, 'num2': 4, 'sum': 6, 'difference': 2})

# Use the function
fn(3, 4)
```

#### Example Output 3
```{'sum': 7, 'difference': 1}```

## 5. Integrating with OpenAI JSON Mode
- If you want to use the OpenAI JSON Mode (which is pretty good btw), you can simply add in ```openai_json_mode = True``` in ```strict_json``` or ```strict_function```
- Note that the model must be one of ```gpt-4-1106-preview``` or ```gpt-3.5-turbo-1106```. We will set it to ```gpt-3.5-turbo-1106``` by default if you provide an invalid model

#### Example Usage
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment',
                                    'Adjectives': 'List of adjectives',
                                    'Words': 'Number of words'},
                    openai_json_mode = True) # Toggle this to True
                                    
print(res)
```

#### Example output
```{'Sentiment': 'Positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 6}```

## 6. Additional Output Field Checks (Advanced)

- You can also specify your own custom check function that will be used to check the output field (which will be in `str`, `int`, `float`, `list` or `dict` format inferred by LLM or specified in `type: <data type>`)
- Ensure that what you are checking for is implied in the output field's description in `output_format` of `strict_json` or `strict_function`
- Your custom check function must take in: `output_field`
- Your custom check function must output: 
    - `requirement` (str): The requirement you are checking for
    - `requirement_met` (bool): Whether condition is met, True or False
    - `action_needed` (str): What needs to be done to meet requirement if requirement_met is False
- If `requirement_met` is False, the `action_needed` message will be used for the `strict_json` error correcting mechanism. Otherwise, the error correcting mechanism will not be triggered
- `action_needed` is used to tell the LLM what it needs to do to meet your requirements (LLM is not able to self-correct without guidance for most cases). Try to be as specific as possible to improve error correction success rate.
- Pass in your custom check function inside `custom_checks` variable of `strict_json` or `strict_function` under the same key as that in `output_format`
- You can add multiple check functions for one variable by putting it inside the same list
- Example custom check function named `hello_world_check` which checks for the presence of hello world
- You can also use information in the variable `check_data` for checks (input via `strict_json` or `strict_function`)

#### Example Custom Check Functions
```python
def hello_world_check(output_field, check_data) -> (str, bool, str):
    ''' Example function 1: Checks whether hello world is present in output_field. '''
    requirement = 'Check whether hello world is present in output field'
    requirement_met = True
    action_needed = ''
    # do a check for requirement of having 'hello'
    if 'hello' not in str(output_field):
        requirement_met = False
        action_needed += 'Add in the word hello into output field, '
    if 'world' not in str(output_field):
        requirement_met = False
        action_needed += 'Add in the word world into output field, '
    return (requirement, requirement_met, action_needed)
```

```python
def function_name_check(output_field, check_data) -> (str, bool, str):
    ''' Example function 2: Checks whether function name is present in output_field
    Uses additional information from the check_data variable of strict_json'''
    function_name = check_data['Function name']
    requirement = f'Check whether {function_name} is present in output field'
    requirement_met = True
    action_needed = ''
    
    # do a check for requirement of having 'myprint'
    if function_name not in str(output_field):
        requirement_met = False
        action_needed += f'Ensure that function name "{function_name}" is used, '
    return (requirement, requirement_met, action_needed)
```

#### Example Usage 1 (in strict_json)
```python
# we can input our custom_checks as a list of check functions, and check_data is the additional information for these check functions
res = strict_json(system_prompt = 'You are a code generator',
                    user_prompt = 'Print out hello world',
                    output_format = {'Thoughts': 'How to do it',
                                    'Python Code': 'Function beginning with def myprint() -> str:'},
                    custom_checks = {'Python Code': [hello_world_check, function_name_check]},
                    check_data = {'Function name:' 'myprint'})
                                    
print(res)
```
#### Example Output 1
`Running check for "Check whether hello world is present in output field" on output field of "Python Code"
Requirement met`


`Running check for "Check whether myprint is present in output field" on output field of "Python Code"
Requirement met`


`{'Thoughts': 'To print out "hello world", use the print() function in Python.',`
`'Python Code': 'def myprint() -> str:\n    return "hello world"'}`

#### Example Usage 2 (in strict_function)

```python
fn = strict_function(fn_description = 'Output code to print hello world in a function named <var1>', 
                     output_format = {'Python code': 'Python function named <var1> to print hello world'},
                     custom_checks = {'Python code': [function_name_check]})

# in runtime of function, we can input what we would want to check in check_data if we are not sure what it will be beforehand
fn('hello world', 'myprint', check_data = {'Function name': 'myprint'})
```

#### Example Output 2

`Running check for "Check whether myprint is present in output field" on output field of "Python code"
Requirement met`

`{'Python code': 'def myprint():\n    print("hello world")'}`

# Future Features:
- Agents with Tool Use
- Conversational Agents
