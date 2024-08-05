# Strict JSON v5.1.2
[UPDATE]: For Agentic Framework, do check out TaskGen (the official Agentic Framework building on StrictJSON). This will make the StrictJSON repo neater and this github will focus on using StrictJSON for LLM Output Parsing
- https://github.com/simbianai/taskgen

### A Strict JSON Framework for LLM Outputs, that fixes problems that json.loads() cannot solve
- Works for JSON outputs with multiple ' or " or { or } or \ or unmatched braces/brackets that may break a json.loads()

### Base Functionalities (see Tutorial.ipynb)
- Ensures LLM outputs into a dictionary based on a JSON format (HUGE: Nested lists and dictionaries now supported)
- Supports `int`, `float`, `str`, `dict`, `list`, `array`, `code`, `Dict[]`, `List[]`, `Enum[]`, `bool` type forcing with LLM-based error correction, as well as LLM-based error correction using `type: ensure <restriction>`, and (advanced) custom user checks using `custom_checks`
- Easy construction of LLM-based functions using ```Function``` (Note: renamed from `strict_function` to keep in line with naming convention of capitalised class groups. `strict_function` still works for legacy support.)
- Easy integration with OpenAI JSON Mode by setting `openai_json_mode = True`
- Exposing of llm variable for `strict_json` and `Function` for easy use of self-defined LLMs
- `AsyncFunction` and `strict_json_async` for async (and faster) processing

### Tutorials and Community Support
- Created: 7 Apr 2023
- Collaborators welcome
- Video tutorial (Ask Me Anything): [https://www.youtube.com/watch?v=L4aytve5v1Q](https://www.youtube.com/watch?v=L4aytve5v1Q)
- Video tutorial: [https://www.youtube.com/watch?v=IjTUKAciTCg](https://www.youtube.com/watch?v=1N-znDTlhNc)
- Discussion Channel (my discord - John's AI Group): [discord.gg/bzp87AHJy5](discord.gg/bzp87AHJy5)

## How do I use this? 
1. Download package via command line ```pip install strictjson```
2. Import the required functions from ```strictjson```
3. Set up the relevant API Keys for your LLM if needed. Refer to ```Tutorial.ipynb``` for how to do it for Jupyter Notebooks.

## How does it work?
- Extract JSON values as a string using a special regex (add delimiters to ```key``` to make ```###key###```) to split keys and values. (New!) Also works for nested datatypes by splitting recursively.
- Uses ```ast.literal_eval``` to best match the extracted output value to a literal (e.g. int, string, dict).
- Ensures that all JSON fields are output by LLM, with optional type checking, if not it will feed in error message to LLM to iteratively correct its generation (default: 3 tries)

# Features:
# 1. Basic Generation

- **system_prompt**: Write in whatever you want the LLM to become. "You are a \<purpose in life\>"
- **user_prompt**: The user input. Later, when we use it as a function, this is the function input
- **output_format**: JSON of output variables in a dictionary, with the key as the output key, and the value as the output description
    - The output keys will be preserved exactly, while the LLM will generate content to match the description of the value as best as possible
- **llm**: The llm you want to use. Takes in `system_prompt` and `user_prompt` and outputs the LLM-generated string

#### Example LLM Definition
```python
def llm(system_prompt: str, user_prompt: str) -> str:
    ''' Here, we use OpenAI for illustration, you can change it to your own LLM '''
    # ensure your LLM imports are all within this function
    from openai import OpenAI
    
    # define your own LLM here
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature = 0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
```

#### Example Usage
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment',
                                    'Adjectives': 'Array of adjectives',
                                    'Words': 'Number of words'},
                    llm = llm)
                                    
print(res)
```

#### Example Output
```{'Sentiment': 'Positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7}```

## 2. Advanced Generation
- More advanced demonstration involving code that would typically break ```json.loads()```

#### Example Usage
```python
res = strict_json(system_prompt = 'You are a code generator, generating code to fulfil a task',
                    user_prompt = 'Given array p, output a function named func_sum to return its sum',
                    output_format = {'Elaboration': 'How you would do it',
                                     'C': 'Code',
                                    'Python': 'Code'},
                    llm = llm)
                                    
print(res)
```

#### Example Output
```{'Elaboration': 'Use a loop to iterate through each element in the array and add it to a running total.', ```

```'C': 'int func_sum(int p[], int size) {\n    int sum = 0;\n    for (int i = 0; i < size; i++) {\n        sum += p[i];\n    }\n    return sum;\n}', ```

```'Python': 'def func_sum(p):\n    sum = 0\n    for num in p:\n        sum += num\n    return sum'}```

## 3. Type forcing output variables
- Generally, ```strict_json``` will infer the data type automatically for you for the output fields
- However, if you would like very specific data types, you can do data forcing using ```type: <data_type>``` at the last part of the output field description
- ```<data_type>``` must be of the form `int`, `float`, `str`, `dict`, `list`, `array`, `code`, `Dict[]`, `List[]`, `Array[]`, `Enum[]`, `bool` for type checking to work
- `code` removes all unicode escape characters that might interfere with normal code running
- The `Enum` and `List` are not case sensitive, so `enum` and `list` works just as well
- For `Enum[list_of_category_names]`, it is best to give an "Other" category in case the LLM fails to classify correctly with the other options.
- If `list` or `List[]` is not formatted correctly in LLM's output, we will correct it by asking the LLM to list out the elements line by line
- For `dict`,  we can further check whether keys are present using `Dict[list_of_key_names]`
- Other types will first be forced by rule-based conversion, any further errors will be fed into LLM's error feedback mechanism
- If `<data_type>` is not the specified data types, it can still be useful to shape the output for the LLM. However, no type checking will be done.
- Note: LLM understands the word `Array` better than `List` since `Array` is the official JSON object type, so in the backend, any type with the word `List` will be converted to `Array`.

### LLM-based checks
- If you would like the LLM to ensure that the type is being met, use `type: ensure <requirement>`
- This will run a LLM to check if the requirement is met. If requirement is not met, the LLM will generate what needs to be done to meet the requirement, which will be fed into the error-correcting loop of `strict_json`

#### Example Usage 1
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment, type: Enum["Pos", "Neg", "Other"]',
                                    'Adjectives': 'Array of adjectives, type: List[str]',
                                    'Words': 'Number of words, type: int',
                                    'In English': 'Whether sentence is in English, type: bool'},
                  llm = llm)
                                    
print(res)
```

#### Example Output 1
```{'Sentiment': 'Pos', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7, 'In English': True}```

#### Example Usage 2
```python
res = strict_json(system_prompt = 'You are an expert at organising birthday parties',
                    user_prompt = 'Give me some information on how to organise a birthday',
                    output_format = {'Famous Quote about Age': 'type: ensure quote contains the word age',
                                    'Lucky draw numbers': '3 numbers from 1-50, type: List[int]',
                                    'Sample venues': 'Describe two venues, type: List[Dict["Venue", "Description"]]'},
                    llm = llm)

print(res)
```

#### Example Output 2
`Using LLM to check "The secret of staying young is to live honestly, eat slowly, and lie about your age. - Lucille Ball" to see if it adheres to "quote contains the word age" Requirement Met: True`


```{'Famous Quote about Age': 'The secret of staying young is to live honestly, eat slowly, and lie about your age. - Lucille Ball',```
```'Lucky draw numbers': [7, 21, 35],```

```'Sample venues': [{'Venue': 'Beachside Resort', 'Description': 'A beautiful resort with stunning views of the beach. Perfect for a summer birthday party.'}, {'Venue': 'Indoor Trampoline Park', 'Description': 'An exciting venue with trampolines and fun activities. Ideal for an active and energetic birthday celebration.'}]}```

## 4. Functions
- Enhances ```strict_json()``` with a function-like interface for repeated use of modular LLM-based functions (or wraps external functions)
- Use angle brackets <> to enclose input variable names. First input variable name to appear in `fn_description` will be first input variable and second to appear will be second input variable. For example, `fn_description = 'Adds up two numbers, <var1> and <var2>'` will result in a function with first input variable `var1` and second input variable `var2`
- (Optional) If you would like greater specificity in your function's input, you can describe the variable after the : in the input variable name, e.g. `<var1: an integer from 10 to 30>`. Here, `var1` is the input variable and `an integer from 10 to 30` is the description.
- (Optional) If your description of the variable is one of `int`, `float`, `str`, `dict`, `list`, `array`, `code`, `Dict[]`, `List[]`, `Array[]`, `Enum[]`, `bool`, we will enforce type checking when generating the function inputs in `get_next_subtask` method of the `Agent` class. Example: `<var1: int>`. Refer to Section 3. Type Forcing Output Variables for details.
- Inputs (primary):
    - **fn_description**: String. Function description to describe process of transforming input variables to output variables. Variables must be enclosed in <> and listed in order of appearance in function input.
        - New feature: If `external_fn` is provided and no `fn_description` is provided, then we will automatically parse out the fn_description based on docstring of `external_fn`. The docstring should contain the names of all compulsory input variables
        - New feature: If `external_fn` is provided and no `output_format` is provided, then we will automatically derive the `output_format` from the function signature
    - **output_format**: Dict. Dictionary containing output variables names and description for each variable.
    
- Inputs (optional):
    - **examples** - Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
    - **external_fn** - Python Function. If defined, instead of using LLM to process the function, we will run the external function. 
        If there are multiple outputs of this function, we will map it to the keys of `output_format` in a one-to-one fashion
    - **fn_name** - String. If provided, this will be the name of the function. Otherwise, if `external_fn` is provided, it will be the name of `external_fn`. Otherwise, we will use LLM to generate a function name from the `fn_description`
    - **kwargs** - Dict. Additional arguments you would like to pass on to the strict_json function
        
- Outputs:
    JSON of output variables in a dictionary (similar to ```strict_json```)
    
#### Example Usage 1 (Description only)
```python
# basic configuration with variable names (in order of appearance in fn_description)
fn = Function(fn_description = 'Output a sentence with <obj> and <entity> in the style of <emotion>', 
                     output_format = {'output': 'sentence'},
                     llm = llm)

# Use the function
fn('ball', 'dog', 'happy') #obj, entity, emotion
```

#### Example Output 1
```{'output': 'The happy dog chased the ball.'}```

#### Example Usage 2 (Examples only)
```python
# Construct the function: infer pattern from just examples without description (here it is multiplication)
fn = Function(fn_description = 'Map <var1> and <var2> to output based on examples', 
                     output_format = {'output': 'final answer'}, 
                     examples = [{'var1': 3, 'var2': 2, 'output': 6}, 
                                 {'var1': 5, 'var2': 3, 'output': 15}, 
                                 {'var1': 7, 'var2': 4, 'output': 28}],
                     llm = llm)

# Use the function
fn(2, 10) #var1, var2
```

#### Example Output 2
```{'output': 20}```

#### Example Usage 3 (Description and Examples)
```python
# Construct the function: description and examples with variable names
# variable names will be referenced in order of appearance in fn_description
fn = Function(fn_description = 'Output the sum and difference of <num1> and <num2>', 
                 output_format = {'sum': 'sum of two numbers', 
                                  'difference': 'absolute difference of two numbers'},
                 examples = {'num1': 2, 'num2': 4, 'sum': 6, 'difference': 2},
                 llm = llm)

# Use the function
fn(3, 4) #num1, num2
```

#### Example Output 3
```{'sum': 7, 'difference': 1}```

#### Example Usage 4 (External Function with automatic inference of fn_description and output_format - Preferred)
```python
# Docstring should provide all input variables, otherwise we will add it in automatically
# We will ignore shared_variables, *args and **kwargs
# No need to define llm in Function for External Functions
from typing import List
def add_number_to_list(num1: int, num_list: List[int], *args, **kwargs) -> List[int]:
    '''Adds num1 to num_list'''
    num_list.append(num1)
    return num_list

fn = Function(external_fn = add_number_to_list)

# Show the processed function docstring
print(str(fn))

# Use the function
fn(3, [2, 4, 5])
```
#### Example Output 5
`Description: Adds <num1: int> to <num_list: list>`

`Input: ['num1', 'num_list']`

`Output: {'num_list': 'Array of numbers'}`

`{'num_list': [2, 4, 5, 3]}`

#### Example Usage 5 (External Function with manually defined fn_description and output_format - Legacy Approach)
```python
def binary_to_decimal(x):
    return int(str(x), 2)

# an external function with a single output variable, with an expressive variable description
fn = Function(fn_description = 'Convert input <x: a binary number in base 2> to base 10', 
            output_format = {'output1': 'x in base 10'},
            external_fn = binary_to_decimal,
            llm = llm)

# Use the function
fn(10) #x
```

#### Example Output 4
```{'output1': 2}```

## 5. Integrating with OpenAI JSON Mode
- If you want to use the OpenAI JSON Mode, you can simply add in ```openai_json_mode = True``` and set ```model = 'gpt-4-1106-preview'``` or ```model = 'gpt-3.5-turbo-1106'``` in ```strict_json``` or ```Function```
- We will set model to ```gpt-3.5-turbo-1106``` by default if you provide an invalid model
- This does not work with the `llm` variable
- Note that type checking does not work with OpenAI JSON Mode

#### Example Usage
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment',
                                    'Adjectives': 'Array of adjectives',
                                    'Words': 'Number of words'},
                    model = 'gpt-3.5-turbo-1106' # Set the model
                    openai_json_mode = True) # Toggle this to True
                                    
print(res)
```

#### Example Output
```{'Sentiment': 'positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 6}```

## 6. Nested Outputs
- StrictJSON supports nested outputs like nested lists and dictionaries

#### Example Input
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': ['Type of Sentiment', 
                                                   'Strength of Sentiment, type: Enum[1, 2, 3, 4, 5]'],
                                    'Adjectives': "Name and Description as separate keys, type: List[Dict['Name', 'Description']]",
                                    'Words': {
                                        'Number of words': 'Word count', 
                                        'Language': {
                                              'English': 'Whether it is English, type: bool',
                                              'Chinese': 'Whether it is Chinese, type: bool'
                                                  },
                                        'Proper Words': 'Whether the words are proper in the native language, type: bool'
                                        }
                                    },
                 llm = llm)

print(res)
```

#### Example Output
`{'Sentiment': ['Positive', 3],`

`'Adjectives': [{'Name': 'beautiful', 'Description': 'pleasing to the senses'}, {'Name': 'sunny', 'Description': 'filled with sunshine'}],`

`'Words':`

`     {'Number of words': 6,`
    
`     'Language': {'English': True, 'Chinese': False},`

`     'Proper Words': True}`
    
`}`

## 7. Return as JSON
- By default, `strict_json` returns a Python Dictionary
- If needed to parse as JSON, simply set `return_as_json=True`
- By default, this is set to `False` in order to return a Python Dictionry

## 8. Async Mode

- `AsyncFunction` and `strict_json_async`
    - These are the async equivalents of `Function` and `strict_json`
    - You will need to define an LLM that can operate in async mode
    - Everything is the same as the sync version of the functions, except you use the `await` keyword when calling `AsyncFunction` and `strict_json_async`
    
    
- Using Async can help do parallel processes simulataneously, resulting in a much faster workflow

#### Example LLM in Async Mode
```python
async def llm_async(system_prompt: str, user_prompt: str):
    ''' Here, we use OpenAI for illustration, you can change it to your own LLM '''
    # ensure your LLM imports are all within this function
    from openai import AsyncOpenAI
    
    # define your own LLM here
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model='gpt-4o-mini',
        temperature = 0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
```

#### Example Input (strict_json_async)
```python
res = await strict_json_async(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment',
                                    'Adjectives': 'Array of adjectives',
                                    'Words': 'Number of words'},
                                     llm = llm_async) # set this to your own LLM

print(res)
```

#### Example Output
`{'Sentiment': 'Positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7}`

#### Example Input (AsyncFunction)
```python
fn =  AsyncFunction(fn_description = 'Output a sentence with <obj> and <entity> in the style of <emotion>', 
                     output_format = {'output': 'sentence'},
                     llm = llm_async) # set this to your own LLM

res = await fn('ball', 'dog', 'happy') #obj, entity, emotion

print(res)
```

#### Example Output
`{'output': 'The dog happily chased the ball.'}`
