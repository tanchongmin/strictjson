from typing import Any, Dict, List, Union, Optional, Annotated, Callable, Awaitable, get_origin, get_args
from pydantic import BaseModel
from datetime import date, datetime, time
import types
from uuid import UUID
from decimal import Decimal
import re
import warnings
import inspect
import yaml

from .helper_functions import convert_schema_to_pydantic

### YAML Parsing Helper Functions ###
def ensure_awaitable(func, name):
    """ Utility function to check if the function is an awaitable coroutine function """
    if func is not None and not inspect.iscoroutinefunction(func):
        raise TypeError(f"{name} must be an awaitable coroutine function")
    
SPECIALS = r'%:#\-\?\{\}\[\],&\*\!\|>\'\"@`'

def quote_special_keys(yaml_text: str) -> str:
    """
    Quote mapping keys that contain special characters and are not already quoted.
    Works for both 'key:' and list item '- key:' lines, without consuming ':'.
    Also escapes inner double quotes when adding outer quotes.
    """

    specials = set(SPECIALS)

    # Plain mapping line: "  key  : ..."
    plain_key_re = re.compile(r'^([ \t]*)(?![ \t]*-\s)([^:\n]+?)([ \t]*):', re.MULTILINE)

    # List-of-maps line: "  - key  : ..."
    list_key_re  = re.compile(r'^([ \t]*-\s+)([^:\n]+?)([ \t]*):', re.MULTILINE)

    def is_already_quoted(s: str) -> bool:
        s = s.strip()
        return len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"')

    def needs_quoting(key: str) -> bool:
        # Trim only the right side so we preserve the user's left spacing in the group
        k = key.rstrip()
        return any(ch in specials for ch in k)

    def wrap_and_escape(key: str) -> str:
        # Preserve original spacing around the textual key
        left_ws = len(key) - len(key.lstrip(' '))
        right_ws = len(key) - len(key.rstrip(' '))
        core = key.strip()

        # If already quoted, keep as-is
        if is_already_quoted(core):
            return key

        if not needs_quoting(core):
            return key

        # Escape backslashes and inner double quotes before wrapping in double quotes
        core_escaped = core.replace('\\', '\\\\').replace('"', r'\"')
        return (' ' * left_ws) + f'"{core_escaped}"' + (' ' * right_ws)

    def repl_list(m: re.Match) -> str:
        indent_dash, key, space = m.groups()
        return f'{indent_dash}{wrap_and_escape(key)}{space}:'

    def repl_plain(m: re.Match) -> str:
        indent, key, space = m.groups()
        return f'{indent}{wrap_and_escape(key)}{space}:'

    # IMPORTANT: do list items first, then plain keys
    out = list_key_re.sub(repl_list, yaml_text)
    out = plain_key_re.sub(repl_plain, out)
    return out
    
def quote_values_with_colons(yaml_text: str) -> str:
    """
    If a mapping value on the same line contains ':' and looks like a plain scalar,
    quote it:  key: a: b  ->  key: "a: b"

    If a mapping line has an inline value AND the next non-empty line is more indented
    (i.e., a block mapping follows), drop the inline scalar:  key: a: b  ->  key:

    This version tightens the matcher so list item headers like '- key:' are handled
    correctly and not misinterpreted.
    """
    lines = yaml_text.splitlines()
    out = []
    i = 0

    def is_quoted(s: str) -> bool:
        s = s.strip()
        return len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'"))

    while i < len(lines):
        line = lines[i]

        # indent, optional "- ", key (unquoted), colon, value (rest of line)
        m = re.match(r'^([ \t]*)(-\s+)?(?!\s*$)([^"\':\n][^:\n]*?):[ \t]*(.*)$', line)
        if not m:
            out.append(line)
            i += 1
            continue

        indent, dash, key, val = m.groups()
        base_indent_len = len(indent)

        # If there's an inline value and the next significant line is more indented,
        # assume user intended a block mapping; drop the inline scalar.
        if val.strip():
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                next_indent_len = len(lines[j]) - len(lines[j].lstrip(' '))
                if next_indent_len > base_indent_len:
                    out.append(f'{indent}{dash or ""}{key}:')
                    i += 1
                    continue

        # Otherwise, if the inline value contains a colon and isn't already quoted
        # and doesn't start a flow collection or block scalar, quote it.
        v = val.strip()
        if (':' in v and not is_quoted(v)
                and not v.startswith(('{', '[', '|', '>', '&', '*', '!'))):
            out.append(f'{indent}{dash or ""}{key}: "{v}"')
        else:
            out.append(line)

        i += 1

    return '\n'.join(out)

# --- BEGIN: schema → YAML template helpers ---
def _python_type_to_placeholder(tp) -> str:
    if tp is str: return "str"
    if tp is int: return "int"
    if tp is float: return "float"
    if tp is bool: return "bool"
    if tp is type(None): return "None"
    if tp is date: return "date"
    if tp is datetime: return "datetime"
    if tp is time: return "time"
    if tp is UUID: return "UUID"
    if tp is Decimal: return "Decimal"
    return tp.__name__ if hasattr(tp, "__name__") else str(tp)

def _with_desc(text: Any, desc: Optional[str]):
    # Only prefix when returning a scalar/list/dict **type string**,
    # not when returning nested dicts (models) or example lists.
    if isinstance(text, str) and desc:
        return f"{desc}, {text}"
    return text

def _annot_to_template(tp, field_desc: Optional[str] = None):
    origin = get_origin(tp)
    args = get_args(tp)

    # Unions: typing.Union[...] or PEP 604 (A | B)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) != len(args)
        if not non_none:
            return _with_desc("Optional[Any]", field_desc)
        if len(non_none) == 1:
            inner = _annot_to_template(non_none[0])
            if isinstance(inner, (dict, list)):
                return inner
            return _with_desc(inner if not has_none else f"Optional[{inner}]", field_desc)
        flat = []
        for a in non_none:
            ar = _annot_to_template(a)
            flat.append(ar if isinstance(ar, str) else "Any")
        s = f"Optional[Union[{', '.join(flat)}]]" if has_none else f"Union[{', '.join(flat)}]"
        return _with_desc(s, field_desc)

    # typing.Annotated[T, ...] → unwrap to T
    if origin is Annotated:
        return _annot_to_template(args[0], field_desc)

    # List[T]
    if origin in (list, List):
        inner = args[0] if args else Any
        inner_repr = _annot_to_template(inner)
        out = [inner_repr] if isinstance(inner_repr, dict) else f"List[{inner_repr}]"
        return _with_desc(out if isinstance(out, str) else out, field_desc)

    # Dict[K, V]
    if origin in (dict, Dict):
        k = _annot_to_template(args[0]) if args else "str"
        v = _annot_to_template(args[1]) if len(args) > 1 else "Any"
        if isinstance(v, (dict, list)):
            v = "Any"  # keep concise
        return _with_desc(f"Dict[{k}, {v}]", field_desc)

    # Nested Pydantic model → map of its fields (v2: FieldInfo in model_fields)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        out = {}
        for fname, f in tp.model_fields.items():
            key = f.alias or fname             # v2
            desc = f.description or None       # v2
            out[key] = _annot_to_template(f.annotation, desc)
        return out

    # Scalar
    return _with_desc(_python_type_to_placeholder(tp), field_desc)

def convert_pydantic_to_yaml(model_cls) -> str:
    if not (isinstance(model_cls, type) and issubclass(model_cls, BaseModel)):
        raise TypeError("model_cls must be a Pydantic BaseModel subclass")
    template_obj = _annot_to_template(model_cls)
    if not isinstance(template_obj, dict):
        template_obj = {"value": template_obj}
    return yaml.dump(template_obj, sort_keys=False)

# --- END: schema to yaml template ---

        
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
        - pydantic_model (BaseModel) = None: pydantic model to use to generate output. If provided, takes priority over output_format.
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

    # get the yaml string
    if output_format is not None:
        yaml_format = yaml.dump(output_format, sort_keys=False)
    else:
        # derive a YAML template from the provided pydantic_model (handles nested refs, arrays, unions)
        yaml_format = convert_pydantic_to_yaml(pydantic_model)

    if debug:
        print("## Concise YAML format used by parse_yaml:", yaml_format, sep = '\n')

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
Make sure you output all the keys in the yaml schema.
If any key contains non-alphanumeric characters, quote the key (e.g., "% of total": 100.0)
Begin output with ```yaml and end output with ```.
    '''

    if retry_prompt:
        retry_prompt = retry_prompt.format(yaml_format = yaml_format)
    else:
        retry_prompt = f'''\nFirst output how you would solve the error (without changing the schema).
Then update the values of this yaml schema:
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
Make sure you output all the keys in the yaml schema.
If any key contains non-alphanumeric characters, quote the key (e.g., "% of total": 100.0)
Ensure your response only contains exactly one block that starts with ```yaml and ends with ```
'''
        
    data, error = None, None

    if type_checking:
        # Convert output format into pydantic
        try:
            # if there no pydantic model defined, then generate from output_format
            if not pydantic_model:
                pydantic_model = convert_schema_to_pydantic(output_format)

            if debug:
                schema_dump = pydantic_model.model_json_schema()
                print("\n\n## Equivalent YAML Schema:", yaml.dump(schema_dump), sep='\n')
    
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

            try:
                data = yaml.safe_load(cleaned_yaml)
            except Exception:
                # attempt to quote special keys (like "% of total")
                cleaned_yaml = quote_special_keys(cleaned_yaml)
                # attempt to remove double colons in the same line
                cleaned_yaml = quote_values_with_colons(cleaned_yaml)
                data = yaml.safe_load(cleaned_yaml)

            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')

        except Exception as e:
            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')
            error = "Parsing of YAML failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)

            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            if accept_kwargs:
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}", **kwargs)
            else:
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}")
        
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
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}", **kwargs)
            else:
                res = llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}")

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
        - pydantic_model (BaseModel) = None: pydantic model to use to generate output. If present, takes priority over output_format
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

    # get the yaml string
    if output_format is not None:
        yaml_format = yaml.dump(output_format, sort_keys=False)
    else:
        # derive a YAML template from the provided pydantic_model (handles nested refs, arrays, unions)
        yaml_format = convert_pydantic_to_yaml(pydantic_model)

    if debug:
        print("## Concise YAML format used by parse_yaml:", yaml_format, sep = '\n')

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
Make sure you output all the keys in the yaml schema.
If any key contains non-alphanumeric characters, quote the key (e.g., "% of total": 100.0)
Begin output with ```yaml and end output with ```.
'''

    if retry_prompt:
        retry_prompt = retry_prompt.format(yaml_format = yaml_format)
    else:
        retry_prompt = f'''\nFirst output how you would solve the error (without changing the schema).
Then update the values of this yaml schema:
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
Make sure you output all the keys in the yaml schema.
If any key contains non-alphanumeric characters, quote the key (e.g., "% of total": 100.0)
Ensure your response only contains exactly one block that starts with ```yaml and ends with ```
'''
        
    data, error = None, None

    if type_checking:
        # Convert output format into pydantic
        try:
            # if there no pydantic model defined, then generate from output_format
            if not pydantic_model:
                pydantic_model = convert_schema_to_pydantic(output_format)
                 
            if debug:
                schema_dump = pydantic_model.model_json_schema()
                print("\n\n## Equivalent YAML Schema:", yaml.dump(schema_dump), sep='\n')
    
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

            try:
                data = yaml.safe_load(cleaned_yaml)
            except Exception:
                # attempt to quote special keys (like "% of total")
                cleaned_yaml = quote_special_keys(cleaned_yaml)
                # attempt to remove double colons in the same line
                cleaned_yaml = quote_values_with_colons(cleaned_yaml)
                data = yaml.safe_load(cleaned_yaml)

            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')

        except Exception as e:
            if debug:
                print("\n\n## Parsed YAML before type checks:", cleaned_yaml, sep = '\n')
            error = "Parsing of YAML failed\n" + str(e)
            if debug:
                print("\n\n## Error: ", error)

            # End if out of tries
            if cur_try == num_tries-1:
                continue

            # feed in the error to the llm and try again
            if accept_kwargs:
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}", **kwargs)
            else:
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}")

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
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}", **kwargs)
            else:
                res = await llm(retry_prompt, f"Incorrectly formatted YAML block: {cleaned_yaml}\nError: {error}")

            if debug:
                print("\n\n## LLM retry response:", res)

            continue

    # if yaml has already been cleaned, return that
    if data and force_return:
        return data
    else:
        return {}