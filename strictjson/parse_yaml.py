from typing import Any, Dict, List, Union, Callable, Awaitable
from pydantic import BaseModel, Field
import pydantic, enum, typing
from datetime import date, datetime, time
from uuid import UUID
from decimal import Decimal
from enum import Enum
import re
import warnings


def parse_yaml(system_prompt: str, 
               user_prompt: str, 
               output_format: dict,
               initial_prompt: str = None, 
               retry_prompt: str = None, 
               type_checking: bool = True, 
               verbose: bool = False, 
               debug: bool = False, 
               num_tries: int = 3, 
               force_return: bool = False,
               llm: Callable[[str, str], str] = None):
    ''' Outputs in the given output_format. Converts to yaml at backend, does type checking with pydantic, and then converts back to dict to user 
    Inputs:
        - system_prompt (str): llm system message
        - user_prompt (str): llm user message
        - output_format (dict): dictionary of keys and values to output
        - initial_prompt (str) = None: prompt to parse yaml
        - retry_prompt (str) = None: prompt to retry parsing of yaml
        - type_checking (bool) = False: whether to do type checking
        - verbose (bool) = False: whether to show the system_prompt and the user_prompt and result of LLM
        - debug (bool) = False: whether or not to print out the intermediate steps like LLM output, parsed yaml, pydantic code
        - num_tries (int) = 3: Number of times to retry generation
        - force_return (bool) = False: If the type checks fail, still return the dictionary if yaml is parsed correctly
        - llm (Callable[[str, str], str]) = None: the llm to use
    Output: 
        - parsed yaml in dictionary form or empty dictionary if unable to parse (dict)
    '''

    import yaml
    if llm is None:
        raise Exception("You need to assign an llm variable that takes in system_prompt and user_prompt as inputs, and outputs the completion")

    # get the yaml string
    yaml_format = yaml.dump(output_format, sort_keys=False)

    # get the configured prompt
    if initial_prompt:
        initial_prompt = initial_prompt.format(yaml_format = yaml_format)
    else:
        initial_prompt = f'''\nUpdate this yaml schema according to datatype (without changing the schema):
```yaml
{yaml_format}
```
Do not use ``` within yaml block. Use | to denote start of coding block, if any.
Begin output with ```yaml.
    '''

    if retry_prompt:
        retry_prompt = retry_prompt.format(yaml_format = yaml_format)
    else:
        retry_prompt = f'''\nFirst output how you would solve the error (without changing the schema).
Then fill this yaml schema according to datatype (begin output with ```yaml):
```yaml
{yaml_format}
```
Do not use ``` within yaml block. Use | to denote start of coding block, if any.
Even if schema does not make sense, just output default values to suit it.
'''
        
    generated_code, data, Yaml_Schema, error = None, None, None, None

    if type_checking:
        # Convert output format into pydantic - Part 1: Getting the code
        try:
            generated_code = convert_schema_to_pydantic(output_format)
            if debug:
                print("\n\n## Generated pydantic schema code:", generated_code, sep='\n')
    
        except Exception as e:
            error = e
            raise Exception(f"Unable to parse output_format into pydantic schema. Check your output_format again.\nError: {error}")
    
        # Convert output format into pydantic - Part 2: Running the code
        try:
            # This code execution is safe - it is a rule-based generation of Pydantic based on the output_format
            # also, we do it in a new namespace with restricted imports and built-ins
            namespace = safe_exec(code = generated_code, allowed_modules = {'typing', 'enum', 'pydantic'})
            Yaml_Schema = namespace["Yaml_Schema"]
        except Exception as e:
            raise Exception(f"Unable to generate pydantic schema.\nError: {e}")
        

    # show the user the system prompt and user prompt if verbose
    if verbose:
        print("\n\n## System Prompt:", system_prompt + initial_prompt)
        print("\n\n## User Prompt:", user_prompt)
        
    # pass it through an llm
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
            pattern = r"```yaml\s*(.*?)\s*```"
            match = re.search(pattern, res, flags=re.DOTALL)
            if match:
                cleaned_yaml = match.group(1)
            else:
                cleaned_yaml = res

            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')
            data = yaml.safe_load(cleaned_yaml)

        except Exception as e:
            error = "Parsing of YAML failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)

            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            res = llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}")

            if debug:
                print("\n\n## LLM retry response:", res)

            continue

        ### Test 2: Perform type checks
        try:
            Yaml_Schema.model_validate(data)
    
            # return the dictionary-processed data
            return data
    
        except Exception as e:
            error = "Type checks failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)
            
            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
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
               output_format: dict,
               initial_prompt: str = None, 
               retry_prompt: str = None, 
               type_checking: bool = True, 
               verbose: bool = False, 
               debug: bool = False, 
               num_tries: int = 3, 
               force_return: bool = False,
               llm: Callable[[str, str], Awaitable[str]] = None):
    ''' Outputs in the given output_format. Converts to yaml at backend, does type checking with pydantic, and then converts back to dict to user 
    Inputs:
        - system_prompt (str): llm system message
        - user_prompt (str): llm user message
        - output_format (dict): dictionary of keys and values to output
        - initial_prompt (str) = None: prompt to parse yaml
        - retry_prompt (str) = None: prompt to retry parsing of yaml
        - type_checking (bool) = False: whether to do type checking
        - verbose (bool) = False: whether to show the system_prompt and the user_prompt and result of LLM
        - debug (bool) = False: whether or not to print out the intermediate steps like LLM output, parsed yaml, pydantic code
        - num_tries (int) = 3: Number of times to retry generation
        - force_return (bool) = False: If the type checks fail, still return the dictionary if yaml is parsed correctly
        - llm (Callable[[str, str], str]) = None: the llm to use
    Output: 
        - parsed yaml in dictionary form or empty dictionary if unable to parse (dict)
    '''

    import yaml
    if llm is None:
        raise Exception("You need to assign an llm variable that takes in system_prompt and user_prompt as inputs, and outputs the completion")

    # get the yaml string
    yaml_format = yaml.dump(output_format, sort_keys=False)

    # get the configured prompt
    if initial_prompt:
        initial_prompt = initial_prompt.format(yaml_format = yaml_format)
    else:
        initial_prompt = f'''\nUpdate this yaml schema according to datatype (without changing the schema):
```yaml
{yaml_format}
```
Do not use ``` within yaml block. Use | to denote start of coding block, if any.
Begin output with ```yaml.
    '''

    if retry_prompt:
        retry_prompt = retry_prompt.format(yaml_format = yaml_format)
    else:
        retry_prompt = f'''\nFirst output how you would solve the error (without changing the schema).
Then fill this yaml schema according to datatype (begin output with ```yaml):
```yaml
{yaml_format}
```
Do not use ``` within yaml block. Use | to denote start of coding block, if any.
Even if schema does not make sense, just output default values to suit it.
'''
        
    generated_code, data, Yaml_Schema, error = None, None, None, None

    if type_checking:
        # Convert output format into pydantic - Part 1: Getting the code
        try:
            generated_code = convert_schema_to_pydantic(output_format)
            if debug:
                print("\n\n## Generated pydantic schema code:", generated_code, sep='\n')
    
        except Exception as e:
            error = e
            raise Exception(f"Unable to parse output_format into pydantic schema. Check your output_format again.\nError: {error}")
    
        # Convert output format into pydantic - Part 2: Running the code
        try:
            # This code execution is safe - it is a rule-based generation of Pydantic based on the output_format
            # also, we do it in a new namespace with restricted imports and built-ins
            namespace = safe_exec(code = generated_code, allowed_modules = {'typing', 'enum', 'pydantic'})
            Yaml_Schema = namespace["Yaml_Schema"]
        except Exception as e:
            raise Exception(f"Unable to generate pydantic schema.\nError: {e}")
        

    # show the user the system prompt and user prompt if verbose
    if verbose:
        print("\n\n## System Prompt:", system_prompt + initial_prompt)
        print("\n\n## User Prompt:", user_prompt)
        
    # pass it through an llm
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
            pattern = r"```yaml\s*(.*?)\s*```"
            match = re.search(pattern, res, flags=re.DOTALL)
            if match:
                cleaned_yaml = match.group(1)
            else:
                cleaned_yaml = res

            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')
            data = yaml.safe_load(cleaned_yaml)

        except Exception as e:
            error = "Parsing of YAML failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)

            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}")

            if debug:
                print("\n\n## LLM retry response:", res)

            continue

        ### Test 2: Perform type checks
        try:
            Yaml_Schema.model_validate(data)
    
            # return the dictionary-processed data
            return data
    
        except Exception as e:
            error = "Type checks failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)
            
            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {res}\nError: {error}")

            if debug:
                print("\n\n## LLM retry response:", res)

            continue

    # if yaml has already been cleaned, return that
    if data and force_return:
        return data
    else:
        return {}
        
def safe_exec(code: str, allowed_modules: set = None) -> dict:
    """
    Executes the provided Python code in a restricted environment where only the modules
    specified in allowed_modules can be imported. It pre-populates the globals with
    certain names (like BaseModel and Field from pydantic, Enum from enum, and typing)
    so that the generated code can reference them without error.
    
    Parameters:
        code (str): The Python code to execute.
        allowed_modules (set): A set of module names allowed for import.
                               If None, defaults to {'pydantic', 'enum', 'typing'}.
    
    Returns:
        dict: The namespace containing the definitions from the executed code.
    
    Raises:
        ImportError: If the executed code tries to import a module that is not allowed.
    """
    import builtins

    if allowed_modules is None:
        allowed_modules = {'pydantic', 'enum', 'typing'}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Allow if the module name exactly matches or starts with one of the allowed modules plus a dot.
        allowed = any(name == mod or name.startswith(mod + ".") for mod in allowed_modules)
        if not allowed:
            raise ImportError(f"Importing module '{name}' is not allowed")
        return __import__(name, globals, locals, fromlist, level)

    # Define a restricted list of safe built-in functions.
    safe_builtin_names = [
        'abs', 'all', 'any', 'bin', 'bool', 'callable', 'chr', 'dict', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'hash', 'hex', 'int', 'isinstance',
        'issubclass', 'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
        'ord', 'pow', 'range', 'repr', 'reversed', 'round', 'set', 'slice',
        'sorted', 'str', 'sum', 'tuple', 'zip'
    ]
    safe_builtins = {name: getattr(builtins, name) for name in safe_builtin_names}
    safe_builtins['__import__'] = safe_import
    safe_builtins['__build_class__'] = builtins.__build_class__

    # Set up the globals dictionary with restricted builtins and __name__.
    safe_globals = {
        '__builtins__': safe_builtins,
        '__name__': '__main__'
    }

    # Pre-populate safe_globals with allowed objects.
    safe_globals['pydantic'] = pydantic
    safe_globals['BaseModel'] = pydantic.BaseModel
    safe_globals['Field'] = pydantic.Field
    
    safe_globals['enum'] = enum
    safe_globals['Enum'] = enum.Enum
    
    safe_globals['typing'] = typing
    # Optionally, you could add specific names from typing if needed:
    safe_globals['Any'] = typing.Any
    safe_globals['Dict'] = typing.Dict
    safe_globals['List'] = typing.List
    safe_globals['Union'] = typing.Union
    safe_globals['Optional'] = typing.Optional
    
    # Add additional dependencies for custom types
    safe_globals['date'] = date
    safe_globals['datetime'] = datetime
    safe_globals['time'] = time
    safe_globals['UUID'] = UUID
    safe_globals['Decimal'] = Decimal

    namespace = safe_globals
    exec(code, namespace)
    return namespace

# --- Type mapping and regex ---
# We want "list" to be parsed as generic only when subscripting is present.
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
def split_outside_brackets(s: str) -> (str, str):
    """
    Splits the string at the last comma that is not inside square brackets.
    Returns a tuple (before, after). If no such comma exists, returns (s, None).
    """
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
    """
    Splits the string on commas that are not inside square brackets.
    For example:
      split_comma_outside_brackets("int, Optional[str]") -> ["int", "Optional[str]"]
      split_comma_outside_brackets("Union[int, str]") -> ["Union[int, str]"]
    """
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

# --- Parsing functions ---
def parse_generic_type(type_str: str) -> str:
    """Recursively parse a generic type string such as 'list[list[int]]', 'Optional[int]', 'Union[list, int, None]' or similar."""
    type_str = type_str.strip()
    if type_str.startswith("type:"):
        type_str = type_str[len("type:"):].strip()
    # Handle Optional types.
    if type_str.startswith("Optional[") and type_str.endswith("]"):
        inner = type_str[len("Optional["):-1].strip()
        return f"Optional[{parse_generic_type(inner)}]"
    # Handle Union types specially.
    if type_str.startswith("Union[") and type_str.endswith("]"):
        inner = type_str[len("Union["):-1].strip()
        parts = split_comma_outside_brackets(inner)
        inner_types = [parse_generic_type(p) for p in parts]
        return f"Union[{', '.join(inner_types)}]"
    # If no generic part, return simple mapping.
    if '[' not in type_str:
        return type_map.get(type_str.lower(), "str")
    # Otherwise, split into base and inner.
    base_end = type_str.index('[')
    base = type_str[:base_end].strip()
    # For generic types, if base is "list" or "array", use "List" (with a capital L).
    if base.lower() in ("list", "array"):
        base_py = "List"
    else:
        base_py = type_map.get(base.lower(), "str")
    # Extract inner content (assumes well-formed brackets)
    inner = type_str[base_end+1:-1].strip()
    parts = split_comma_outside_brackets(inner)
    inner_types = [parse_generic_type(p) for p in parts]
    return f"{base_py}[{', '.join(inner_types)}]"

def determine_type(val: str) -> (str, bool):
    """Return (parsed_type, True) if val is recognized; else ("str", False)."""
    val = val.strip()
    # Remove a leading "type:" if present.
    if val.lower().startswith("type:"):
        val = val[len("type:"):].strip()
    # Check for Enum types and preserve them.
    if val.startswith("Enum[") and val.endswith("]"):
        return (val, True)
    # Special-case: if it's exactly "list" or "array", return List[Any]
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
    return ("str", False)

def determine_field_type_and_desc(val: str) -> (str, str):
    """
    Determines if the input string contains type information at the end.
    If so, splits at the last comma outside brackets. If the second part
    starts with "Enum[" (or is a valid type), returns that as the type and the
    preceding part as description.
    Otherwise, if the whole string is a recognized type, returns (type, None),
    else returns ("Any", full val).
    """
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
    """Remove quotes and replace spaces with underscores for valid identifiers."""
    return key.replace('"', "").replace("'", "").replace(" ", "_")

# --- Processing schema values ---
def merge_dicts(dicts: List[Dict]) -> Dict:
    """Merge a list of dicts (taking the first occurrence for each key)."""
    merged = {}
    for d in dicts:
        for k, v in d.items():
            if k not in merged:
                merged[k] = v
    return merged

def process_schema_value(key: str, value: Any, parent: str) -> (str, str, Dict):
    """
    Process a schema value:
      - If string, determine its type and description.
      - If dict, assume nested model.
      - If list:
          * If all dicts, merge and create a custom model.
          * Otherwise, use the type of the first element.
    Returns (field_type, description, extra_models).
    """
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
        # Use parent's name to disambiguate nested models.
        new_model_name = "".join(word.capitalize() for word in (parent + "_" + sanitize_key(key)).split("_"))
        extras[new_model_name] = value
        return (new_model_name, None, extras)
    elif isinstance(value, str):
        t, desc = determine_field_type_and_desc(value)
        return (t, desc, {})
    else:
        return ("str", None, {})

def maybe_forward_ref(field_type: str, custom_models: set) -> str:
    """Wrap field_type in quotes if it refers to a custom model."""
    if field_type in custom_models:
        return f'"{field_type}"'
    if field_type.startswith("List[") and field_type.endswith("]"):
        inner = field_type[5:-1].strip()
        if inner in custom_models:
            return f'List["{inner}"]'
    return field_type

# --- Code Generation ---
def convert_schema_to_pydantic(schema: Dict, main_model: str = "Yaml_Schema") -> str:
    """
    Convert a schema dict into generated Pydantic code.
    """
    extras = {}  # Mapping of model names to schema dicts or enum definitions.
    field_lines = []
    
    # Process top-level fields.
    for key, val in schema.items():
        attr = sanitize_key(key)
        alias = key if key != attr else None
        # First, process the field value.
        ftype, desc, extra = process_schema_value(key, val, main_model)
        extras.update(extra)
        # If the determined type starts with "Enum[", process it as an Enum.
        if isinstance(val, str) and (ftype.startswith("Enum[") or (desc is None and ftype.startswith("Enum["))):
            # Use the field name to generate the enum class name.
            enum_name = attr + "Enum"
            # Remove the "Enum[" and trailing "]" from ftype.
            content = ftype[len("Enum["):-1].strip()
            opts = [p.strip() for p in content.split(",")]
            enum_lines = [f"class {enum_name}(Enum):"]
            for opt in opts:
                # If the option is quoted, treat it as a string.
                if (opt.startswith("'") and opt.endswith("'")) or (opt.startswith('"') and opt.endswith('"')):
                    opt_val = opt[1:-1]
                    numeric = False
                else:
                    opt_val = opt
                    try:
                        _ = int(opt_val)
                        numeric = True
                    except ValueError:
                        try:
                            _ = float(opt_val)
                            numeric = True
                        except ValueError:
                            numeric = False
                member = re.sub(r'\W+', '_', opt_val).upper()
                if re.match(r'^\d', member):
                    member = '_' + member
                if numeric:
                    enum_lines.append(f'    {member} = {opt_val}')
                else:
                    enum_lines.append(f'    {member} = "{opt_val}"')
            extras[enum_name] = "\n".join(enum_lines)
            ftype = enum_name
            desc = None

        # Build the field line.
        field_def = f"{attr}: {ftype}"
        opts = []
        if alias and alias != attr:
            opts.append(f'alias="{alias}"')
        if desc:
            opts.append(f'description="{desc}"')
        if opts:
            field_def += " = Field(..., " + ", ".join(opts) + ")"
        field_lines.append("    " + field_def)
    
    # --- Recursively collect nested extras ---
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

    # Build extra models.
    extra_lines = []
    custom_models = {name for name, s in extras.items() if isinstance(s, dict)}
    for name in sorted(extras.keys(), key=lambda n: n.count('_'), reverse=True):
        content = extras[name]
        if isinstance(content, str):
            extra_lines.append(content)
            extra_lines.append("")
        elif isinstance(content, dict):
            extra_lines.append(f"class {name}(BaseModel):")
            for sub_key, sub_val in content.items():
                sub_attr = sanitize_key(sub_key)
                if isinstance(sub_val, (list, dict)):
                    stype, sdesc, _ = process_schema_value(sub_key, sub_val, name)
                    stype = maybe_forward_ref(stype, custom_models)
                elif isinstance(sub_val, str):
                    stype, sdesc = determine_field_type_and_desc(sub_val)
                    stype = maybe_forward_ref(stype, custom_models)
                else:
                    stype, sdesc = ("str", None)
                sub_def = f"{sub_attr}: {stype}"
                sub_opts = []
                if sub_key != sub_attr:
                    sub_opts.append(f'alias="{sub_key}"')
                if sdesc:
                    sub_opts.append(f'description="{sdesc}"')
                if sub_opts:
                    sub_def += " = Field(..., " + ", ".join(sub_opts) + ")"
                extra_lines.append("    " + sub_def)
            extra_lines.append("")
    
    # Build main model.
    main_lines = [f"class {main_model}(BaseModel):"]
    for fl in field_lines:
        for cname in custom_models:
            fl = re.sub(rf'(:\s+){cname}(\s*=?\s*)', rf'\1"{cname}"\2', fl)
        main_lines.append(fl)
    main_lines.append("")
    main_lines.append("    class Config:")
    main_lines.append("        populate_by_name = True")
    
    # Build import lines.
    import_lines = [
        "from pydantic import BaseModel, Field",
        "from enum import Enum"
    ]
    typing_imports = set(["Dict"])
    if any("List[" in l for l in field_lines) or any("List[" in l for l in extra_lines):
        typing_imports.add("List")
    if any("Union[" in l for l in field_lines) or any("Union[" in l for l in extra_lines):
        typing_imports.add("Union")
    if any("Any" in l for l in field_lines) or any("Any" in l for l in extra_lines):
        typing_imports.add("Any")
    if any("Optional[" in l for l in field_lines) or any("Optional[" in l for l in extra_lines):
        typing_imports.add("Optional")
    import_lines.append("from typing import " + ", ".join(sorted(typing_imports)))
    
    out = "\n".join(import_lines) + "\n\n"
    out += "\n".join(extra_lines) + "\n"
    out += "\n".join(main_lines)
    return out