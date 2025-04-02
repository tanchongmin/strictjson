from typing import Any, Dict, List, Union, Optional, Callable, Awaitable
from pydantic import BaseModel, Field, create_model
from datetime import date, datetime, time
import uuid
from uuid import UUID
from decimal import Decimal
from enum import Enum
import re
import warnings
import json
import inspect

### Helper Functions ###
def ensure_awaitable(func, name):
    """ Utility function to check if the function is an awaitable coroutine function """
    if func is not None and not inspect.iscoroutinefunction(func):
        raise TypeError(f"{name} must be an awaitable coroutine function")

### Main Functions ###

def parse_yaml(system_prompt: str, 
               user_prompt: str, 
               output_format: dict = None,
               pydantic_model: BaseModel = None,
               initial_prompt: str = None, 
               retry_prompt: str = None, 
               type_checking: bool = True, 
               verbose: bool = False, 
               debug: bool = False, 
               num_tries: int = 3, 
               force_return: bool = False,
               llm: Callable[[str, str], str] = None,
               **kwargs):
    ''' Outputs in the given output_format. Converts to yaml at backend, does type checking with pydantic, and then converts back to dict to user 
    Inputs:
        - system_prompt (str): llm system message
        - user_prompt (str): llm user message
        - output_format (dict) = None: dictionary of keys and values to output
        - pydantic_model (BaseModel) = None: pydantic model to use to generate output. Goes straight to LLM provider for structured outputs
        - initial_prompt (str) = None: prompt to parse yaml
        - retry_prompt (str) = None: prompt to retry parsing of yaml
        - type_checking (bool) = False: whether to do type checking
        - verbose (bool) = False: whether to show the system_prompt and the user_prompt and result of LLM
        - debug (bool) = False: whether or not to print out the intermediate steps like LLM output, parsed yaml, pydantic code
        - num_tries (int) = 3: Number of times to retry generation
        - force_return (bool) = False: If the type checks fail, still return the dictionary if yaml is parsed correctly
        - llm (Callable[[str, str], str], **kwargs) = None: the llm to use
        - **kwargs: Any other variables to pass into llm
    Output: 
        - parsed yaml in dictionary form or empty dictionary if unable to parse (dict)
    '''

    import yaml
    # add in more yaml safe_load null options
    yaml.SafeLoader.add_implicit_resolver(
                    u'tag:yaml.org,2002:null',
                    re.compile(r'^(?:~|null|Null|NULL|None)$'),
                    list("~nN")
                )
    
    if llm is None:
        raise Exception("You need to assign an llm variable that takes in system_prompt and user_prompt as inputs, and outputs the completion")

    # check whether llm accepts kwargs
    sig = inspect.signature(llm)
    accept_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

    if not output_format and not pydantic_model:
        raise Exception("You need to either have an output_format or a pydantic_model specified")

    # Does a direct pass into LLM if there is a pydantic_model defined
    if pydantic_model:
        res = llm(system_prompt, user_prompt, pydantic_model = pydantic_model)
        return json.loads(res)

    # get the yaml string
    yaml_format = yaml.dump(output_format, sort_keys=False)

    # get the configured prompt
    if initial_prompt:
        initial_prompt = initial_prompt.format(yaml_format = yaml_format)
    else:
        initial_prompt = f'''\nUpdate the values of this yaml schema (without changing the schema):
```yaml
{yaml_format}
```
List format:
<key_name>:
 - <item_1>
 - <item_2>

Dictionary format:
<key_name>:
  <key_name>: <value>
  <key_name>: <value>

Multi-line output format:
<key_name>: |
  <line 1>
  <line 2>
Output the appropriate values that meets the required datatypes specified.
Begin output with ```yaml and end output with ```.
    '''

    if retry_prompt:
        retry_prompt = retry_prompt.format(yaml_format = yaml_format)
    else:
        retry_prompt = f'''\nFirst output how you would solve the error (without changing the schema).
Then update the values of this yaml schema (begin output with ```yaml and end output with ```):
```yaml
{yaml_format}
```
List format:
<key_name>:
 - <item_1>
 - <item_2>

Dictionary format:
<key_name>:
  <key_name>: <value>
  <key_name>: <value>

Multi-line output format:
<key_name>: |
  <line 1>
  <line 2>
Output the appropriate values that meets the required datatypes specified.
Even if schema does not make sense, just output default values to suit it.
'''
        
    generated_code, data, pydantic_model, error = None, None, None, None

    if type_checking:
        # Convert output format into pydantic
        try:
            pydantic_model = convert_schema_to_pydantic(output_format)
            kwargs["pydantic_model"] = pydantic_model
            if debug:
                print("\n\n## Generated YAML Schema:", pydantic_model.schema(), sep='\n')
    
        except Exception as e:
            error = e
            raise Exception(f"Unable to parse output_format into pydantic schema. Check your output_format again.\nError: {error}")

    # show the user the system prompt and user prompt if verbose
    if verbose:
        print("\n\n## System Prompt:", system_prompt + initial_prompt)
        print("\n\n## User Prompt:", user_prompt)
        
    # pass it through an llm (add in kwargs if llm accepts)
    if accept_kwargs:
        res = llm(system_prompt + initial_prompt, user_prompt, **kwargs)
    else:
        res = llm(system_prompt + initial_prompt, user_prompt)

    # show user the llm result if verbose or debug
    if verbose or debug:
        print("\n\n## LLM Result:", res)

    for cur_try in range(num_tries):
        if cur_try > 0:
            my_warning = f"LLM Parsing failed at attempt {cur_try}.\nRetrying..."
            warnings.warn(my_warning, UserWarning)
        ### Test 1: Test whether YAML can be parsed
        try:
            pattern = r"```yaml\s*(.*)\s*```"
            match = re.search(pattern, res, flags=re.DOTALL)
            if match:
                cleaned_yaml = match.group(1)
            else:
                cleaned_yaml = res

            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')
            try:
                data = yaml.safe_load(cleaned_yaml)
            except Exception as e:
                # replace backslash single quotes with double single quotes for better parsing by yaml.safe_load
                cleaned_yaml = cleaned_yaml.replace("\'", "''")
                data = yaml.safe_load(cleaned_yaml)

        except Exception as e:
            error = "Parsing of YAML failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)

            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            if accept_kwargs:
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}", **kwargs)
            else:
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}")
        
            if debug:
                print("\n\n## LLM retry response:", res)

            continue

        ### Test 2: Perform type checks
        try:
            validated_data = pydantic_model.model_validate(data)
    
            # return the dictionary-processed data
            return validated_data.model_dump(by_alias=True)
    
        except Exception as e:
            error = "Type checks failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)
            
            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            if accept_kwargs:
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}", **kwargs)
            else:
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}")

            if debug:
                print("\n\n## LLM retry response:", res)

            continue

    # if yaml has already been cleaned, return that
    if data and force_return:
        return data
    else:
        return {}

async def parse_yaml_async(system_prompt: str, 
               user_prompt: str, 
               output_format: dict = None,
               pydantic_model: BaseModel = None,
               initial_prompt: str = None, 
               retry_prompt: str = None, 
               type_checking: bool = True, 
               verbose: bool = False, 
               debug: bool = False, 
               num_tries: int = 3, 
               force_return: bool = False,
               llm: Callable[[str, str], Awaitable[str]] = None,
               **kwargs):
    ''' Outputs in the given output_format. Converts to yaml at backend, does type checking with pydantic, and then converts back to dict to user 
    Inputs:
        - system_prompt (str): llm system message
        - user_prompt (str): llm user message
        - output_format (dict) = None: dictionary of keys and values to output
        - pydantic_model (BaseModel) = None: pydantic model to use to generate output. Goes straight to LLM provider for structured outputs
        - initial_prompt (str) = None: prompt to parse yaml
        - retry_prompt (str) = None: prompt to retry parsing of yaml
        - type_checking (bool) = False: whether to do type checking
        - verbose (bool) = False: whether to show the system_prompt and the user_prompt and result of LLM
        - debug (bool) = False: whether or not to print out the intermediate steps like LLM output, parsed yaml, pydantic code
        - num_tries (int) = 3: Number of times to retry generation
        - force_return (bool) = False: If the type checks fail, still return the dictionary if yaml is parsed correctly
        - llm (Callable[[str, str], Awaitable[str]], **kwargs) = None: the llm to use
        - **kwargs: Any other variables to pass into llm
    Output: 
        - parsed yaml in dictionary form or empty dictionary if unable to parse (dict)
    '''

    import yaml
    # add in more yaml safe_load null options
    yaml.SafeLoader.add_implicit_resolver(
                    u'tag:yaml.org,2002:null',
                    re.compile(r'^(?:~|null|Null|NULL|None)$'),
                    list("~nN")
                )
    
    if llm is None:
        raise Exception("You need to assign an llm variable that takes in system_prompt and user_prompt as inputs, and outputs the completion")

    # make sure llm is awaitable
    else:
        ensure_awaitable(llm, 'llm')

    # check whether llm accepts kwargs
    sig = inspect.signature(llm)
    accept_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

    if not output_format and not pydantic_model:
        raise Exception("You need to either have an output_format or a pydantic_model specified")

    # Does a direct pass into LLM if there is a pydantic_model defined
    if pydantic_model:
        res = await llm(system_prompt, user_prompt, pydantic_model = pydantic_model)
        return json.loads(res)

    # get the yaml string
    yaml_format = yaml.dump(output_format, sort_keys=False)

    # get the configured prompt
    if initial_prompt:
        initial_prompt = initial_prompt.format(yaml_format = yaml_format)
    else:
        initial_prompt = f'''\nUpdate the values of this yaml schema (without changing the schema):
```yaml
{yaml_format}
```
List format:
<key_name>:
 - <item_1>
 - <item_2>

Dictionary format:
<key_name>:
  <key_name>: <value>
  <key_name>: <value>

Multi-line output format:
<key_name>: |
  <line 1>
  <line 2>
Output the appropriate values that meets the required datatypes specified.
Begin output with ```yaml and end output with ```.
    '''

    if retry_prompt:
        retry_prompt = retry_prompt.format(yaml_format = yaml_format)
    else:
        retry_prompt = f'''\nFirst output how you would solve the error (without changing the schema).
Then update the values of this yaml schema (begin output with ```yaml and end output with ```):
```yaml
{yaml_format}
```
List format:
<key_name>:
 - <item_1>
 - <item_2>

Dictionary format:
<key_name>:
  <key_name>: <value>
  <key_name>: <value>

Multi-line output format:
<key_name>: |
  <line 1>
  <line 2>
Output the appropriate values that meets the required datatypes specified.
Even if schema does not make sense, just output default values to suit it.
'''
        
    generated_code, data, pydantic_model, error = None, None, None, None

    if type_checking:
        # Convert output format into pydantic
        try:
            pydantic_model = convert_schema_to_pydantic(output_format)
            kwargs["pydantic_model"] = pydantic_model
            if debug:
                print("\n\n## Generated YAML Schema:", pydantic_model.schema(), sep='\n')
    
        except Exception as e:
            error = e
            raise Exception(f"Unable to parse output_format into pydantic schema. Check your output_format again.\nError: {error}")

    # show the user the system prompt and user prompt if verbose
    if verbose:
        print("\n\n## System Prompt:", system_prompt + initial_prompt)
        print("\n\n## User Prompt:", user_prompt)

    if accept_kwargs:
        res = await llm(system_prompt + initial_prompt, user_prompt, **kwargs)
    else:
        res = await llm(system_prompt + initial_prompt, user_prompt)

    # show user the llm result if verbose or debug
    if verbose or debug:
        print("\n\n## LLM Result:", res)
        
    for cur_try in range(num_tries):
        if cur_try > 0:
            my_warning = f"LLM Parsing failed at attempt {cur_try}.\nRetrying..."
            warnings.warn(my_warning, UserWarning)
        ### Test 1: Test whether YAML can be parsed
        try:
            pattern = r"```yaml\s*(.*)\s*```"
            match = re.search(pattern, res, flags=re.DOTALL)
            if match:
                cleaned_yaml = match.group(1)
            else:
                cleaned_yaml = res

            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')
            try:
                data = yaml.safe_load(cleaned_yaml)
            except Exception as e:
                # replace backslash single quotes with double single quotes for better parsing by yaml.safe_load
                cleaned_yaml = cleaned_yaml.replace("\'", "''")
                data = yaml.safe_load(cleaned_yaml)

        except Exception as e:
            error = "Parsing of YAML failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)

            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            if accept_kwargs:
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}", **kwargs)
            else:
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}")

            if debug:
                print("\n\n## LLM retry response:", res)

            continue

        ### Test 2: Perform type checks
        try:
            validated_data = pydantic_model.model_validate(data)
    
            # return the dictionary-processed data
            return validated_data.model_dump(by_alias=True)
    
        except Exception as e:
            error = "Type checks failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)
            
            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            if accept_kwargs:
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}", **kwargs)
            else:
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}")

            if debug:
                print("\n\n## LLM retry response:", res)

            continue

    # if yaml has already been cleaned, return that
    if data and force_return:
        return data
    else:
        return {}

#####################################
#### Schema -> Pydantic Function ####
#####################################

import re
from typing import Any, List, Optional, Union, Dict
from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel, Field, create_model
from enum import Enum

# --- Type mapping and regex ---
type_map = {
    "int": "int",
    "integer": "int",
    "float": "float",
    "double": "float",
    "str": "str",
    "string": "str",
    "bool": "bool",
    "boolean": "bool",
    # For generic parsing we override these:
    "list": "list",   
    "array": "list",
    "dict": "dict",
    "object": "dict",
    "date": "date",
    "datetime": "datetime",
    "time": "time",
    "uuid": "UUID",
    "decimal": "Decimal",
    "any": "Any",
    "none": "None"
}

simple_type_regex = re.compile(
    r'^(?:type:\s*)?(int|integer|float|double|str|string|bool|boolean|list|array|dict|object|date|datetime|time|uuid|decimal|any|none)$',
    re.IGNORECASE
)

# --- Helper functions ---
def split_outside_brackets(s: str) -> (str, Optional[str]):
    depth = 0
    last_comma_index = -1
    for i, c in enumerate(s):
        if c == '[':
            depth += 1
        elif c == ']':
            depth = max(depth - 1, 0)
        elif c == ',' and depth == 0:
            last_comma_index = i
    if last_comma_index != -1:
        return s[:last_comma_index].strip(), s[last_comma_index+1:].strip()
    else:
        return s.strip(), None

def split_comma_outside_brackets(s: str) -> List[str]:
    parts = []
    current = []
    depth = 0
    for c in s:
        if c == '[':
            depth += 1
            current.append(c)
        elif c == ']':
            depth -= 1
            current.append(c)
        elif c == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
    if current:
        parts.append(''.join(current).strip())
    return parts

def parse_generic_type(type_str: str) -> str:
    """
    Recursively parses a generic type string such as 'list[list[int]]', 'Optional[int]', or
    'Union[list, int, None]' and returns a normalized type string.
    """
    type_str = type_str.strip()
    if type_str.startswith("type:"):
        type_str = type_str[len("type:"):].strip()
    # <<-- New guard for enums: if the type is an Enum, return as is.
    if type_str.startswith("Enum[") and type_str.endswith("]"):
        return type_str
    # >>--
    if type_str.startswith("Optional[") and type_str.endswith("]"):
        inner = type_str[len("Optional["):-1].strip()
        return f"Optional[{parse_generic_type(inner)}]"
    if type_str.startswith("Union[") and type_str.endswith("]"):
        inner = type_str[len("Union["):-1].strip()
        parts = split_comma_outside_brackets(inner)
        inner_types = [parse_generic_type(p) for p in parts]
        return f"Union[{', '.join(inner_types)}]"
    if '[' not in type_str:
        return type_map.get(type_str.lower(), "Any")
    base_end = type_str.index('[')
    base = type_str[:base_end].strip()
    if base.lower() in ("list", "array"):
        base_py = "List"
    else:
        base_py = type_map.get(base.lower(), "Any")
    inner = type_str[base_end+1:-1].strip()
    parts = split_comma_outside_brackets(inner)
    inner_types = [parse_generic_type(p) for p in parts]
    return f"{base_py}[{', '.join(inner_types)}]"

def determine_type(val: str) -> (str, bool):
    val = val.strip()
    if val.lower().startswith("type:"):
        val = val[len("type:"):].strip()
    if val.startswith("Enum[") and val.endswith("]"):
        return (val, True)
    if val.lower() in ("list", "array"):
        return ("List[Any]", True)
    if simple_type_regex.match(val):
        return (type_map[val.lower()], True)
    if '[' in val and ']' in val:
        try:
            parsed = parse_generic_type(val)
            return (parsed, True)
        except Exception:
            pass
    return ("Any", False)

def determine_field_type_and_desc(val: str) -> (str, Optional[str]):
    description_candidate, possible_type = split_outside_brackets(val)
    if possible_type is not None:
        if possible_type.startswith("Enum["):
            return (possible_type, description_candidate)
        t, recognized = determine_type(possible_type)
        if recognized:
            return (t, description_candidate)
    t, recognized = determine_type(val)
    if recognized:
        return (t, None)
    return ("Any", val.strip())

def sanitize_key(key: str) -> str:
    return key.replace('"', "").replace("'", "").replace(" ", "_")

def merge_dicts(dicts: List[Dict]) -> Dict:
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if k not in merged:
                merged[k] = v
    return merged

def process_schema_value(key: str, value: Any, parent: str) -> (str, Optional[str], Dict):
    extras = {}
    if isinstance(value, list):
        if not value:
            return ("List[Any]", None, {})
        if all(isinstance(el, dict) for el in value):
            new_model_name = "".join(word.capitalize() for word in (parent + "_" + sanitize_key(key)).split("_"))
            extras[new_model_name] = merge_dicts(value)
            return (f"List[{new_model_name}]", None, extras)
        else:
            first = value[0]
            if isinstance(first, str):
                t, desc = determine_field_type_and_desc(first)
                return (f"List[{t}]", desc, {})
            else:
                return ("List[Any]", None, {})
    elif isinstance(value, dict):
        new_model_name = "".join(word.capitalize() for word in (parent + "_" + sanitize_key(key)).split("_"))
        extras[new_model_name] = value
        return (new_model_name, None, extras)
    elif isinstance(value, str):
        t, desc = determine_field_type_and_desc(value)
        return (t, desc, {})
    else:
        return ("Any", None, {})

def convert_schema_to_pydantic(schema: Dict, main_model: str = "pydantic_model") -> Any:
    """
    Dynamically converts a schema dictionary into Pydantic models.
    """
    extras: Dict[str, Any] = {}
    main_fields: Dict[str, Any] = {}
    
    # Dictionary to cache generated Enum types.
    generated_enums: Dict[str, Any] = {}

    # return enum values as the value itself rather than as an Enum instance
    class Config:
        use_enum_values = True

    # Process top-level fields.
    for key, val in schema.items():
        attr = sanitize_key(key)
        alias = key if key != attr else None
        if isinstance(val, str):
            ftype_str, desc = determine_field_type_and_desc(val)
            extra = {}
        else:
            ftype_str, desc, extra = process_schema_value(key, val, main_model)
        extras.update(extra)

        # Special handling for top-level enum definitions.
        if isinstance(val, str) and ftype_str.startswith("Enum[") and ftype_str.endswith("]"):
            enum_name = attr + "Enum"
            content = ftype_str[len("Enum["):-1].strip()
            opts = [p.strip() for p in content.split(",")]
            enum_members = {}
            for opt in opts:
                if ((opt.startswith("'") and opt.endswith("'")) or 
                    (opt.startswith('"') and opt.endswith('"'))):
                    opt_val = opt[1:-1]
                    numeric = False
                else:
                    opt_val = opt
                    try:
                        int(opt_val)
                        numeric = True
                    except ValueError:
                        try:
                            float(opt_val)
                            numeric = True
                        except ValueError:
                            numeric = False
                member = re.sub(r'\W+', '_', opt_val).upper()
                if re.match(r'^\d', member):
                    member = '_' + member
                enum_members[member] = int(opt_val) if numeric else opt_val
            ftype = Enum(enum_name, enum_members)
            desc = None
        else:
            ftype = ftype_str

        field_params = {}
        if alias and alias != attr:
            field_params["alias"] = alias
        if desc:
            field_params["description"] = desc
        main_fields[attr] = (ftype, Field(..., **field_params))
    
    # Recursively collect nested extra models.
    changed = True
    while changed:
        changed = False
        for model_name, content in list(extras.items()):
            if isinstance(content, dict):
                for sub_key, sub_val in content.items():
                    if isinstance(sub_val, (list, dict)):
                        _, _, sub_extra = process_schema_value(sub_key, sub_val, model_name)
                        for k, v in sub_extra.items():
                            if k not in extras:
                                extras[k] = v
                                changed = True

    def parse_type_string(type_ann: Any) -> Any:
        """
        Resolves a type annotation (which may be a string) into an actual Python type.
        """
        if not isinstance(type_ann, str):
            return type_ann
        ts = type_ann.strip()
        if ts in extra_models:
            return extra_models[ts]
        lower = ts.lower()
        simple_mapping = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "dict": dict,
            "list": list,
            "any": Any,
            "none": type(None),
            "date": date,
            "datetime": datetime,
            "time": time,
            "decimal": Decimal,
            "uuid": UUID,
        }
        if lower in simple_mapping:
            if lower in ("list", "array"):
                return List[Any]
            return simple_mapping[lower]
        if ts.startswith("Optional[") and ts.endswith("]"):
            inner = ts[len("Optional["):-1].strip()
            return Optional[parse_type_string(inner)]
        if ts.startswith("Union[") and ts.endswith("]"):
            inner = ts[len("Union["):-1].strip()
            parts = split_comma_outside_brackets(inner)
            return Union[tuple(parse_type_string(part) for part in parts)]
        if ts.startswith("List[") and ts.endswith("]"):
            inner = ts[5:-1].strip()
            return List[parse_type_string(inner)]
        if ts.lower().startswith("dict[") and ts.endswith("]"):
            inner = ts[ts.find("[")+1:-1].strip()
            parts = split_comma_outside_brackets(inner)
            if len(parts) == 2:
                key_type = parse_type_string(parts[0])
                value_type = parse_type_string(parts[1])
                return Dict[key_type, value_type]
            else:
                return dict
        # <<-- New check for nested enums.
        if ts.startswith("Enum[") and ts.endswith("]"):
            if ts in generated_enums:
                return generated_enums[ts]
            content = ts[len("Enum["):-1].strip()
            opts = [p.strip() for p in content.split(",")]
            enum_members = {}
            for opt in opts:
                if ((opt.startswith("'") and opt.endswith("'")) or 
                    (opt.startswith('"') and opt.endswith('"'))):
                    opt_val = opt[1:-1]
                    numeric = False
                else:
                    opt_val = opt
                    try:
                        int(opt_val)
                        numeric = True
                    except ValueError:
                        try:
                            float(opt_val)
                            numeric = True
                        except ValueError:
                            numeric = False
                member = re.sub(r'\W+', '_', opt_val).upper()
                if re.match(r'^\d', member):
                    member = '_' + member
                enum_members[member] = int(opt_val) if numeric else opt_val
            enum_name = "GeneratedEnum" + str(len(generated_enums) + 1)
            enum_type = Enum(enum_name, enum_members)
            generated_enums[ts] = enum_type
            return enum_type
        # >>--
        if ts in extra_models:
            return extra_models[ts]
        return Any

    extra_models: Dict[str, Any] = {}
    # Create extra models using create_model.
    for name, content in extras.items():
        if isinstance(content, dict):
            fields = {}
            for sub_key, sub_val in content.items():
                sub_attr = sanitize_key(sub_key)
                if isinstance(sub_val, (list, dict)):
                    stype_str, sdesc, _ = process_schema_value(sub_key, sub_val, name)
                elif isinstance(sub_val, str):
                    stype_str, sdesc = determine_field_type_and_desc(sub_val)
                else:
                    stype_str, sdesc = ("Any", None)
                sub_params = {}
                if sub_key != sub_attr:
                    sub_params["alias"] = sub_key
                if sdesc:
                    sub_params["description"] = sdesc
                fields[sub_attr] = (parse_type_string(stype_str), Field(..., **sub_params))
            extra_models[name] = create_model(name, __config__=Config, **fields)

    resolved_main_fields = {}
    for field_name, (ftype, field_def) in main_fields.items():
        resolved_type = parse_type_string(ftype)
        resolved_main_fields[field_name] = (resolved_type, field_def)
    
    MainModel = create_model(main_model, __config__=Config, **resolved_main_fields)
    
    for model in extra_models.values():
        model.update_forward_refs()
    MainModel.update_forward_refs()
    
    return MainModel
