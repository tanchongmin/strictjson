# StrictJSON v6.1.2 - A Structured Output Framework for LLM Outputs

## New Functionalities (see Tutorial - parse_yaml.ipynb)
#### Why YAML?
- YAML is much more concise than JSON, and avoids a lot of problems faced with quotations and brackets
- YAML also outputs code blocks much better with multi-line literal-block scalar (|), and the code following it can be totally free of any unnecessary backslash escape characters as it is not in a string
- LLMs now are quite good at interpreting YAML than when this repo was first started

#### How it works
- See `Tutorial - parse_yaml.ipynb` for more information.
- Parses the LLM output as a YAML, and converts it to dict
- Uses concise `output_format` to save tokens
- Converts `output_format` into pydantic schema automatically, and uses pydantic to validate output
- Able to process datatypes: `int`, `float`, `str`, `bool`, `list`, `dict`, `date`, `datetime`, `time`, `UUID`, `Decimal`
- Able to process: `None`, `Any`, `Union`, `Optional`
- Default datatype when not specified is `Any`
- Error correction of up to `num_tries` times (default: 3)
- More streamlined and works for multiple models such as:
    - Claude 3.5 Sonnet
    - Claude 3.7 Sonnet
    - gpt-o3-mini
    - gpt-o1-mini
    - gpt-4o-mini
    - gpt-4o
    - Meta Llama 3.3 70B
    - Meta Llama 3.2 90B (Note: Smaller versions of Llama 3.2 do not work well with YAML)
    - Meta Llama 3.1 70B (Note: Smaller versions of Llama 3.1 do not work well with YAML)
    - DeepSeek-V3
    - DeepSeek-R1
    - QwW 32B
    - Gemini 2.0 Flash
    - Gemini 2.0 Flash-Lite

#### Future Plans for YAML Parsing
- Due to its versatility and better type checking with Pydantic, `parse_yaml` will now be the main focus for development
- `strict_json` will still be around for legacy compatibility

#### Example LLM Definition
```python
def llm(system_prompt: str, user_prompt: str, **kwargs) -> str:
    ''' Here, we use OpenAI for illustration, you can change it to your own local/remote LLM '''
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
parse_yaml(system_prompt = "Give me 5 names on a topic", 
           user_prompt = "weather",
           output_format = {"Names": "Great sounding names, List[str]",
                            "Meanings": "Name and meaning, dict", 
                            "Chinese Meanings": "Name and meaning in chinese, dict",
                            "Lucky Name or Number": "List[Union[int, str]]",
                            "Code": "Python code to generate 5 names"},
           llm = llm)
```

#### Example Output
```
{'Names': ['Aurora', 'Zephyr', 'Nimbus', 'Solstice', 'Tempest'],
 'Meanings': {'Aurora': 'Dawn',
  'Zephyr': 'Gentle breeze',
  'Nimbus': 'Rain cloud',
  'Solstice': 'Sun standing still',
  'Tempest': 'Violent storm'},
 'Chinese Meanings': {'Aurora': '曙光',
  'Zephyr': '微风',
  'Nimbus': '雨云',
  'Solstice': '至日',
  'Tempest': '暴风'},
 'Lucky Name or Number': [7, '13', 3, 'Lucky', 21],
 'Code': 'import random\n\ndef generate_weather_names():\n    names = ["Aurora", "Zephyr", "Nimbus", "Solstice", "Tempest"]\n    return random.sample(names, 5)\n\nprint(generate_weather_names())'}
```

## (Optional) Easy interface with Structured Output parser from your favourite LLM provider!
In the rare event `parse_yaml` fails to generate valid YAML for your use case, you can also use the Structured Output parser directly from your favourite LLM provider.

#### Example LLM Definition to use Structured Outputs natively from LLM provider
```python
def llm(system_prompt: str, user_prompt: str, **kwargs) -> str:
    ''' Here, we use OpenAI for illustration, you can change it to your own LLM '''
    # ensure your LLM imports are all within this function
    from openai import OpenAI

    client = OpenAI()
    params = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    }
    
    # Only add 'response_format' if a pydantic_model is provided.
    if kwargs.get("pydantic_model") is not None:
        params["response_format"] = kwargs["pydantic_model"]

        print("For debugging purposes, this is the json schema for the Pydantic Model:")
        print(kwargs["pydantic_model"].model_json_schema())
    
    response = client.beta.chat.completions.parse(**params)
    return response.choices[0].message.content
```

#### Method 1: Using the pydantic model automatically generated via output_format
```python
parse_yaml(system_prompt = "You are a helpful assistent",
    user_prompt = "Generate a birthday event for Alex",
    output_format = {"name": "str",
                     "date": "str",
                     "participants": "only male names, list[str]"},
                     llm = llm)
```

#### Method 2: Using the pydantic model specified in `parse_yaml` input
```python
from pydantic import BaseModel, Field

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str] = Field(..., description = "only male names")

parse_yaml(system_prompt = "You are a helpful assistent",
    user_prompt = "Generate a birthday event for Alex", 
    pydantic_model = CalendarEvent,
    llm = llm)
```

---

For Agentic Framework, do check out AgentJo (the official Agentic Framework building on StrictJSON). This will make the StrictJSON repo neater and this github will focus on using StrictJSON for LLM Output Parsing
- https://github.com/tanchongmin/agentjo

--- 

## How do I use this? 
1. Download package via command line ```pip install strictjson```
2. Import the required functions from ```strictjson```
   
### Tutorials and Community Support
- Created: 7 Apr 2023
- Collaborators welcome
- Discussion Channel (my discord - John's AI Group): [discord.gg/bzp87AHJy5](discord.gg/bzp87AHJy5)
- Video for `parse_yaml` coming soon
- Videos for `strict_json`:
    - Video tutorial (Ask Me Anything): [https://www.youtube.com/watch?v=L4aytve5v1Q](https://www.youtube.com/watch?v=L4aytve5v1Q)
    - Video tutorial: [https://www.youtube.com/watch?v=IjTUKAciTCg](https://www.youtube.com/watch?v=1N-znDTlhNc)

### Base Functionalities (see Tutorial - strict_json.ipynb)
- Ensures LLM outputs into a dictionary based on a JSON format (HUGE: Nested lists and dictionaries now supported)
- Works for JSON outputs with multiple ' or " or { or } or \ or unmatched braces/brackets that may break a json.loads()
- Supports `int`, `float`, `str`, `dict`, `list`, `array`, `code`, `Dict[]`, `List[]`, `Enum[]`, `bool` type forcing with LLM-based error correction, as well as LLM-based error correction using `type: ensure <restriction>`, and (advanced) custom user checks using `custom_checks`
- Easy construction of LLM-based functions using ```Function``` (Note: renamed from `strict_function` to keep in line with naming convention of capitalised class groups. `strict_function` still works for legacy support.)
- Easy integration with OpenAI JSON Mode by setting `openai_json_mode = True`
- Exposing of llm variable for `strict_json` and `Function` for easy use of self-defined LLMs
- `AsyncFunction` and `strict_json_async` for async (and faster) processing

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

## 4. Integrating with OpenAI JSON Mode
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

## 5. Nested Outputs
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

## 6. Return as JSON
- By default, `strict_json` returns a Python Dictionary
- If needed to parse as JSON, simply set `return_as_json=True`
- By default, this is set to `False` in order to return a Python Dictionry

## 7. Async Mode

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