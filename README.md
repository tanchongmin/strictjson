# Strict JSON v4.1.0
[UPDATE]: For Agentic Framework, do check out TaskGen (the official Agentic Framework building on StrictJSON). This will make the StrictJSON repo neater and this github will focus on using StrictJSON for LLM Output Parsing
- https://github.com/simbianai/taskgen

### A Strict JSON Framework for LLM Outputs, that fixes problems that json.loads() cannot solve
- Works for JSON outputs with multiple ' or " or { or } or \ or unmatched braces/brackets that may break a json.loads()

### Base Functionalities (see Tutorial.ipynb)
- Ensures LLM outputs into a dictionary based on a JSON format (HUGE: Nested lists and dictionaries now supported)
- Supports `int`, `float`, `str`, `dict`, `list`, `Dict[]`, `List[]`, `Enum[]`, `bool` type forcing with LLM-based error correction, as well as LLM-based error correction using `type: ensure <restriction>`, and (advanced) custom user checks using `custom_checks`
- Easy construction of LLM-based functions using ```Function``` (Note: renamed from `strict_function` to keep in line with naming convention of capitalised class groups. `strict_function` still works for legacy support.)
- Easy integration with OpenAI JSON Mode by setting `openai_json_mode = True`
- Exposing of llm variable for `strict_json` and `Function` for easy use of self-defined LLMs

### Tutorials and Community Support
- Created: 7 Apr 2023
- Collaborators welcome
- Video tutorial (Ask Me Anything): [https://www.youtube.com/watch?v=L4aytve5v1Q](https://www.youtube.com/watch?v=L4aytve5v1Q)
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

#### Example Output
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

#### Example Output
```{'Elaboration': 'To calculate the sum of an array, we can iterate through each element of the array and add it to a running total.', ```

```'C': 'int func_sum(int p[], int size) {\n    int sum = 0;\n    for (int i = 0; i < size; i++) {\n        sum += p[i];\n    }\n    return sum;\n}', ```

```'Python': 'def func_sum(p):\n    sum = 0\n    for num in p:\n        sum += num\n    return sum'}```

## 3. Type forcing
- Generally, ```strict_json``` will infer the data type automatically for you for the output fields
- However, if you would like very specific data types, you can do data forcing using ```type: <data_type>``` at the last part of the output field description
- ```<data_type>``` must be of the form `int`, `float`, `str`, `dict`, `list`, `Dict[]`, `List[]`, `Enum[]`, `bool` for type checking to work
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
                                    'Words': 'Number of words, type: int',
                                    'In English': 'Whether sentence is in English, type: bool'})
                                    
print(res)
```

#### Example Output 1
```{'Sentiment': 'Pos', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7, 'In English': True}```

#### Example Usage 2
```python
res = strict_json(system_prompt = 'You are an expert at organising birthday parties',
                    user_prompt = 'Give me some information on how to organise a birthday',
                    output_format = {'Famous Quote': 'quote with name, type: ensure quote contains the word age',
                                    'Lucky draw numbers': '3 numbers from 1-50, type: List[int]',
                                    'Sample venues': 'Describe two venues, type: List[Dict["Venue", "Description"]]'})

print(res)
```

#### Example Output 2
`Using LLM to check "The secret of staying young is to live honestly, eat slowly, and lie about your age. - Lucille Ball" to see if it adheres to "quote contains the word age" Requirement Met: True`

```{'Famous Quote': 'The secret of staying young is to live honestly, eat slowly, and lie about your age. - Lucille Ball',```
```'Lucky draw numbers': [7, 21, 35],```

```'Sample venues': [{'Venue': 'Beachside Resort', 'Description': 'A beautiful resort with stunning views of the beach. Perfect for a summer birthday party.'}, {'Venue': 'Indoor Trampoline Park', 'Description': 'An exciting venue with trampolines and fun activities. Ideal for an active and energetic birthday celebration.'}]}```

# 4. Functions
- Enhances ```strict_json()``` with a function-like interface for repeated use of modular LLM-based functions
- Use angle brackets <> to enclose input variable names. First input variable name to appear in `fn_description` will be first input variable and second to appear will be second input variable. For example, `fn_description = 'Adds up two numbers, <var1> and <var2>'` will result in a function with first input variable `var1` and second input variable `var2`

- Inputs (primary):
    - **fn_description**: String. Function description to describe process of transforming input variables to output variables. Variables must be enclosed in <> and listed in order of appearance in function input.
    - **output_format**: Dict. Dictionary containing output variables names and description for each variable.
    
- Inputs (optional):
    - **examples** - Dict or List[Dict]. Examples in Dictionary form with the input and output variables (list if more than one)
    - **kwargs** - Dict. Additional arguments you would like to pass on to the strict_json function
        
- Outputs:
    JSON of output variables in a dictionary (similar to ```strict_json```)
    
#### Example Usage 1 (Description only)
```python
# basic configuration with variable names (in order of appearance in fn_description)
fn = Function(fn_description = 'Output a sentence with <obj> and <entity> in the style of <emotion>', 
                     output_format = {'output': 'sentence'})

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
                                 {'var1': 7, 'var2': 4, 'output': 28}])

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
                 examples = {'num1': 2, 'num2': 4, 'sum': 6, 'difference': 2})

# Use the function
fn(3, 4) #num1, num2
```

#### Example Output 3
```{'sum': 7, 'difference': 1}```

## 5. Integrating with your own LLM
- StrictJSON has native support for OpenAI LLMs (you can put the LLM API parameters inside `strict_json` or `Function` directly)
- If your LLM is not from OpenAI, it is really easy to integrate with your own LLM
- Simply pass your custom LLM function inside the `llm` parameter of `strict_json` or `Function`
    - Inputs:
        - system_prompt: String. Write in whatever you want the LLM to become. e.g. "You are a \<purpose in life\>"
        - user_prompt: String. The user input. Later, when we use it as a function, this is the function input
    - Output:
        - res: String. The response of the LLM call

#### Example Custom LLM
```python
def llm(system_prompt: str, user_prompt: str):
    ''' Here, we use OpenAI for illustration, you can change it to your own LLM '''
    # ensure your LLM imports are all within this function
    from openai import OpenAI
    
    # define your own LLM here
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        temperature = 0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content
```

#### Example Usage with `strict_json`
```python
res = strict_json(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful and sunny day',
                    output_format = {'Sentiment': 'Type of Sentiment',
                                    'Adjectives': 'List of adjectives',
                                    'Words': 'Number of words'},
                                     llm = llm) # set this to your own LLM

print(res)
```

#### Example Output
```{'Sentiment': 'Positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 7}```

## 6. Integrating with OpenAI JSON Mode
- If you want to use the OpenAI JSON Mode (which is pretty good btw), you can simply add in ```openai_json_mode = True``` in ```strict_json``` or ```Function```
- Note that the model must be one of ```gpt-4-1106-preview``` or ```gpt-3.5-turbo-1106```. We will set it to ```gpt-3.5-turbo-1106``` by default if you provide an invalid model
- Note that type checking does not work with OpenAI JSON Mode

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

#### Example Output
```{'Sentiment': 'Positive', 'Adjectives': ['beautiful', 'sunny'], 'Words': 6}```

## 7. Nested Outputs
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
                                    })

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