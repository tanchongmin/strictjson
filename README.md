# Strict JSON v2
A Strict JSON Framework for LLM Outputs, that fixes problems that json.loads() cannot solve
- Updated: 3 Jan 2024
- Created: 28 Oct 2023
- Video tutorial: https://www.youtube.com/watch?v=IjTUKAciTCg

- Works for JSON outputs with ' or " or \ or { or } or unmatched braces/brackets that may break a json.loads()

## How do I install this?

1. Download entire directory and go to root folder
2. Download python version 3.11 (https://www.python.org/downloads/) - should work for later versions as well, but this was tested on python 3.11
3. pip install -r requirements.txt

## How do I use this?
1. Replace ```<YOUR API KEY HERE>``` in ```os.environ['OPENAI_API_KEY'] = '<YOUR API KEY HERE>'``` with your own OpenAI API key (https://platform.openai.com/account/api-keys)
2. Copy and paste ```strict_text``` and ```strict_function``` from Strict_JSON_v2.ipynb
3. Use the functions as needed

## How does it work?
- Extract JSON values as a string using a special regex (add delimiters to ```key``` to make ```###key###```) to split keys and values
- Uses ```ast.literal_eval``` to best match the string to a literal (e.g. int, string, dict) [Set ```literal_eval = False``` to preserve output as string]
- Ensures that all JSON fields are output by LLM, if not it will feed in error message to LLM to generate better (default: 2 tries)

# Features:
## Basic generation
- **system_prompt**: Write in whatever you want the LLM to become. "You are a \<purpose in life\>"
- **user_prompt**: The user input. Later, when we use it as a function, this is the function input
- **output_format**: JSON format with the key as the output key, and the value as the output description
    - The output keys will be preserved exactly, while GPT will generate content to match the description of the value as best as possible
 
#### Example Usage
```python
res = strict_text(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful day',
                    output_format = {"Sentiment": "Type of Sentiment",
                                    "Tense": "Type of Tense"})
                                    
print(res)
```

#### Example output
```{'Sentiment': 'Positive', 'Tense': 'Present'}```

## Advanced Generation
- More advanced demonstration involving code and multiple generation that would typically break ```json.loads()```

#### Example Usage
```python
res = strict_text(system_prompt = 'You are a code generator, generating code to fulfil a task',
                    user_prompt = 'Sum all elements in a given array p',
                    output_format = {"Elaboration": "How you would do it",
                                     "C": "Code in C",
                                    "Python": "Code in Python"})
                                    
print(res)
```

#### Example output
```{'Elaboration': 'To sum all elements in a given array, you can iterate through each element of the array and keep adding them to a running total.', ```

```'C': 'int sum = 0;\\nfor (int i = 0; i < n; i++) {\\n    sum += p[i];\\n}', ```

```'Python': 'sum = 0\\nfor i in range(len(p)):\\n    sum += p[i]'}```

# Strict JSON Functions

## Overview
- Enhances ```strict_text()``` with a function-like interface for repeated use of modular LLM-based functions
- Inputs (compulsory):
    - **fn_description** - Function description to describe process of transforming input variables to output variables
    - **output_format** - Dictionary containing output variables names and description for each variable. There must be at least one output variable
- Inputs (optional):
    - **examples** - Examples in Dictionary form with the input and output variables (list if more than one)
    - **input_type** - Dictionary containing input variable names and mapping function for variables (need not contain all variables)
    - **output_type** - Dictionary containing output variable names and mapping function for variables (need not contain all variables)
    - **kwargs** - Additional arguments you would like to pass on to the ```strict_text``` function
        
        
- Outputs:
    Dictionary of all the output variables (similar to ```strict_text```)
    
#### Example Usage 1 (Description only)
```python
# Construct the function: var1 will be first input variable, var2 will be second input variable and so on
fn = strict_function(fn_description = 'Output a sentence with words var1 and var2 in the style of var3', 
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

#### Example Usage 3 (Description, Examples and Types)
```python
# Construct the function: var1 will be first input variable, var2 will be second input variable and so on
fn = strict_function(fn_description = 'Output the sum and difference of var1 and var2', 
                 output_format = {'sum': 'sum of two numbers', 'difference': 'absolute difference of two numbers'}, 
                 examples = {'var1': 2, 'var2': 4, 'sum': 6, 'difference': '2'}, 
                 input_type = {'var1': int, 'var2': int},           # optional
                 output_type = {'sum': int, 'difference': str})     # optional

# Use the function
fn(3, 4)
```

#### Example Output 3
```{'sum': 7, 'difference': '1'}```

~ ~ ~ ~ ~

# Future Features:
- Agents with Tool Use
- Conversational Agents
