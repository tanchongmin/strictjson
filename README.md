# stricttext
A Strict JSON Framework for LLM Outputs, that fixes problems with json.loads() dannot solve
- Strict JSON v2
- 28 Oct 2023

- Better than vanilla Strict JSON if you want to output ' or " or \ that may break a json.loads()
## Key Guideline: Bare Minimum, Functional Concept
- "Fit everything into a string, because it works"
- You will get everything back as a string, and you can then convert it to int, float, code, array up to your liking
- With strict_text, you can get any kind of answers including those with lots of ' or " or { or } or \
- You don't even need to match brackets { or quotation marks ' in the json fields for this to work
- Fewer features than vanilla Strict JSON (such as list-based constraining, dynamic inputs), but you can always just type it out in system prompt yourself

## Overall Open-ended generation (advanced)
- More advanced demonstration involving code and multiple generation that would typically break ```json.loads()```

- **system_prompt**: Write in whatever you want GPT to become. "You are a \<purpose in life\>"
- **user_prompt**: The user input. Later, when we use it as a function, this is the function input
- **output_format**: JSON format with the key as the output key, and the value as the output description
    - The output keys will be preserved exactly, while GPT will generate content to match the description of the value as best as possible

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
```{'Elaboration': 'To sum all elements in a given array, you can iterate through each element of the array and add it to a running total.', 'C': 'int sumArray(int p[], int size) {\\n    int sum = 0;\\n    for (int i = 0; i < size; i++) {\\n        sum += p[i];\\n    }\\n    return sum;\\n}', 'Python': 'def sum_array(p):\\n    sum = 0\\n    for num in p:\\n        sum += num\\n    return sum'}```

~ ~ ~ ~ ~

# strictjson (Original)
A Strict JSON Framework for LLM Outputs
- By John Tan Chong Min
- 3 Jul 2023
- Video tutorial: https://www.youtube.com/watch?v=A6sIh-lmApk

# Introduction
- Ever wanted LLMs to generate exactly what you want and nothing more?
- Want LLMs to do classification easily by choosing from a list of categories?
- Don't like the extra baggage of OpenAI Functions and LangChain?
- Then Strict JSON Framework is for you.
- TODO: Integrate with the SymbolicAI framework - https://github.com/Xpitfire/symbolicai

# How it works
- Input your desired output JSON format
- Strict JSON Framework will always return LLM output in your desired JSON format (returns as JSON, not text)
- It prompts GPT iterately via self-correcting rule-based error messages, and gets the desired output after a few iterations
- If it fails to get it after the iteration limit (default: 2), it outputs an empty JSON

# Features:
## Overall Open-ended generation
- **system_prompt**: Write in whatever you want GPT to become. "You are a \<purpose in life\>"
- **user_prompt**: The user input. Later, when we use it as a function, this is the function input
- **output_format**: JSON format with the key as the output key, and the value as the output description
    - The output keys will be preserved exactly, while GPT will generate content to match the description of the value as best as possible
 
#### Example Usage
```python
res = strict_output(system_prompt = 'You are a classifier',
                    user_prompt = 'It is a beautiful day',
                    output_format = {"Sentiment": "Type of Sentiment",
                                    "Tense": "Type of Tense"})
                                    
print(res)
```

#### Example output
```{'Sentiment': 'Positive', 'Tense': 'Present'}```
 
## List-based constraining of outputs

- You can constrain the output of a field by using a list of categories. Then GPT will treat it as a classification problem and return one of the categories
    - Example input text: "I am so elated!"
    - Example output_format: {"Sentiment": ["happy", "sad", "neutral"]}
    - Example output: {"Sentiment": "happy"}
 
## List-based label constraining of output

- You can also constrain the output of a field to label names, by defining the list in the following format {Label Name}: {Label Description}
    - Example input text: "I am so elated!"
    - Example output format: {"Sentiment": ["A: happy", "B: sad", "C: neutral"]}
    - Example output: {"Sentiment": "A"}
 
# Dynamic output format
- Used when we want to constrain the output in a largely fixed format but allow for some flexibility in some areas
- Flexible areas are enclosed with <> alongside areas which are fixed
    - Example: "\<entity\> bit my \<entity\>" means that we want GPT to replace the two \<entity\> tags, but preserve "bit my" exactly
    - Example output: "Fish bit my finger"
- <> can also be applied to the keys of json, but we will not be doing strict output checks on those fields since it will be dynamically generated
- When <> is applied to the output key, GPT can generate the key name according to context

## Chain-of-thought prompting via output format

- You can also perform chain-of-thought prompting by ordering the json fields in the right way
- Example
    - Generate broad plan, and then condition on broad plan to generate detailed plan
    - Can do so by just specifying "Broad Plan" as the first output field, and "Detailed Plan" as the next output field (sequence matters!)
    - We can also prompt the model for thoughts, action, observation (ReAct framework) as part of json output
    - We can also prompt the model for reflection (RefleXion framework), and even combine the two together!
 
## Handling Input as a List
- In order to save tokens, we may want to process multiple input items using the same output_format schema
- We can then pass in a list into user_prompt to get the function to output a list of json
- There will be one json in the output for each element of the input list

## Future Features:
- LLM + Rules-based Adaptive Functions
- Dynamic Tool Use
- Agents with Tool Use
