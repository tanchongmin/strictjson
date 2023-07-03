# strictjson
A Strict JSON Framework for LLM Outputs

- Ever wanted LLMs to generate exactly what you want and nothing more?
- Want LLMs to do classification easily by choosing from a list of categories?
- Don't like the extra baggage of OpenAI Functions and LangChain?

- Then Strict JSON Framework is for you.

- Featuring:
## Overall Open-ended generation
- **system_prompt**: Write in whatever you want GPT to become. "You are a \<purpose in life\>"
- **user_prompt**: The user input. Later, when we use it as a function, this is the function input
- **output_format**: JSON format with the key as the output key, and the value as the output description
    - The output keys will be preserved exactly, while GPT will generate content to match the description of the value as best as possible
 
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
    - Day planner has much better output if broad plan is generated first before detailed plan
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
