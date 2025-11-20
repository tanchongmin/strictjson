#####################################
#### Schema -> Pydantic Function ####
#####################################

import re
import uuid
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, create_model

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
    "list": "List[Any]",
    "array": "List[Any]",
    "dict": "Dict[str, Any]",
    "object": "Dict[str, Any]",
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

def split_pipe_outside_brackets(s: str) -> List[str]:
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
        elif c == '|' and depth == 0:
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

    # --- NEW: PEP 604 support: only handle top-level '|'
    if '|' in type_str and _brackets_balanced(type_str):
        parts = split_pipe_outside_brackets(type_str)
        if len(parts) > 1:  # <-- only treat true top-level unions
            none_markers = {"none", "nonetype", "null"}
            non_none = [p for p in parts if p.strip().lower() not in none_markers and p.strip() != "None"]
            has_none = len(non_none) != len(parts)
            if not non_none:
                return "Optional[Any]"
            if len(non_none) == 1:
                inner = parse_generic_type(non_none[0])
                return f"Optional[{inner}]" if has_none else inner
            inner_types = ", ".join(parse_generic_type(p) for p in non_none)
            return f"Optional[Union[{inner_types}]]" if has_none else f"Union[{inner_types}]"
    # fall through for nested cases like List[str | int]
    
    if type_str.startswith("Optional[") and type_str.endswith("]"):
        inner = type_str[len("Optional["):-1].strip()
        return f"Optional[{parse_generic_type(inner)}]"
    if type_str.startswith("Union[") and type_str.endswith("]"):
        inner = type_str[len("Union["):-1].strip()
        parts = split_comma_outside_brackets(inner)
        inner_types = [parse_generic_type(p) for p in parts]
        return f"Union[{', '.join(inner_types)}]"
    if '[' not in type_str:
        # Strict: only known primitive or special types allowed
        lower = type_str.lower()
        if lower not in type_map:
            raise ValueError(
                f"Invalid or unrecognized type: '{type_str}'. "
                f"Expected one of {list(type_map.keys())} or a generic like list[...]"
            )
        return type_map[lower]
    base_end = type_str.index('[')
    base = type_str[:base_end].strip()
    lower_base = base.lower()
    if lower_base in ("list", "array"):
        base_py = "List"
    elif lower_base in ("dict", "object"):
        base_py = "Dict"
    else:
        if lower_base not in type_map:
            raise ValueError(
                f"Invalid or unrecognized type: '{base}'. "
                f"Expected one of {list(type_map.keys())} or a generic like list[...]"
            )
        base_py = type_map[lower_base]
               
    inner = type_str[base_end+1:-1].strip()
    parts = split_comma_outside_brackets(inner)
    inner_types = [parse_generic_type(p) for p in parts]
    return f"{base_py}[{', '.join(inner_types)}]"

def _brackets_balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == '[': depth += 1
        elif ch == ']':
            depth -= 1
            if depth < 0: return False
    return depth == 0
         
def determine_type(val: str) -> (str, bool):
    val = val.strip()
    if not _brackets_balanced(val):
        # raise error if brackets are unbalanced
        raise ValueError(f"Unbalanced brackets in: {val}")
    if val.lower().startswith("type:"):
        val = val[len("type:"):].strip()
    if val.startswith("Enum[") and val.endswith("]"):
        return (val, True)
    if val.lower() in ("list", "array"):
        return ("List[Any]", True)
    if val.lower() in ("dict", "object"):
        return ("Dict[str, Any]", True)
    if simple_type_regex.match(val):
        return (type_map[val.lower()], True)
    
    # If it *looks* like a generic annotation, parse and let errors bubble up
    if '[' in val or ']' in val:
        parsed = parse_generic_type(val)  # let ValueError propagate
        return (parsed, True)
    if '|' in val:
        parsed = parse_generic_type(val)
        return (parsed, True)
                
    # if it doesn't match anything, return Any
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
    # --- cleanup old models ---
    for name in list(globals()):
        if name.startswith("pydantic_model"):
            del globals()[name]
            
    # --- end cleanup ---
    if main_model == "pydantic_model":
        main_model = f"pydantic_model_{uuid.uuid4().hex}"
        
    extras: Dict[str, Any] = {}
    main_fields: Dict[str, Any] = {}
    
    # Dictionary to cache generated Enum types.
    generated_enums: Dict[str, Any] = {}

    # return enum values as the value itself rather than as an Enum instance
    class BaseWithCfg(BaseModel):
        model_config = ConfigDict(use_enum_values=True, populate_by_name=True, extra="forbid")

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
            # Prefer v2-style explicit aliases
            field_params["validation_alias"] = alias
            field_params["serialization_alias"] = alias
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

        # --- NEW: PEP 604 union resolver (top-level only)
        if '|' in ts and _brackets_balanced(ts):
            pipe_parts = split_pipe_outside_brackets(ts)
            if len(pipe_parts) > 1:  # <-- only if the '|' is top-level
                none_markers = {"none", "nonetype", "null"}
                non_none = [p for p in pipe_parts if p.strip().lower() not in none_markers and p.strip() != "None"]
                has_none = len(non_none) != len(pipe_parts)
                if not non_none:
                    return Optional[Any]
                resolved = [parse_type_string(p) for p in non_none]
                if len(resolved) == 1:
                    return Optional[resolved[0]] if has_none else resolved[0]
                union_type = Union[tuple(resolved)]
                return Optional[union_type] if has_none else union_type
        # fall through so nested cases like List[str | int] resolve when List[...] branch parses the inner
        
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
            parts = split_comma_outside_brackets(inner)
            if len(parts) != 1 or parts[0] == "":
                raise ValueError(
                    f"Invalid list type: '{ts}'. "
                    f"A list must specify exactly one inner type, e.g., List[str]"
                )
            return List[parse_type_string(parts[0])]
        if ts.lower().startswith("dict[") and ts.endswith("]"):
            inner = ts[ts.find("[")+1:-1].strip()
            parts = split_comma_outside_brackets(inner)
        
            # Enforce exactly two type parameters for dict
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid dict type: '{ts}'. "
                    f"A dict must specify both key and value types, e.g., Dict[str, int]"
                )
        
            key_type = parse_type_string(parts[0])
            value_type = parse_type_string(parts[1])
            return Dict[key_type, value_type]
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
                    sub_params["validation_alias"] = sub_key
                    sub_params["serialization_alias"] = sub_key
                if sdesc:
                    sub_params["description"] = sdesc
                fields[sub_attr] = (parse_type_string(stype_str), Field(..., **sub_params))
            extra_models[name] = create_model(name, __base__=BaseWithCfg, **fields)

    resolved_main_fields = {}
    for field_name, (ftype, field_def) in main_fields.items():
        resolved_type = parse_type_string(ftype)
        resolved_main_fields[field_name] = (resolved_type, field_def)
    
    MainModel = create_model(main_model, __base__=BaseWithCfg, **resolved_main_fields)
    
    for model in extra_models.values():
        model.model_rebuild()
    MainModel.model_rebuild()
    
    return MainModel

## Another version to get it ready to OpenAI's limited structured output
import re
from typing import Any, Dict

def convert_schema_to_openai_pydantic(schema: Dict, main_model: str = "pydantic_model") -> Any:
    """
    Like convert_schema_to_pydantic, but preprocesses the schema so that the
    resulting Pydantic model's JSON schema is more compatible with OpenAI
    structured outputs.

    Key behaviors:
    - Bare list types become list[str]:
        * "list", "List", "array", "type: list" → "list[str]"
        * "some desc, list" → "some desc, list[str]"
    - 'any' / 'Any' are normalized to 'str' so we never emit an unconstrained
      'Any' in the JSON schema (which leads to items: {}).
    - 'none' / 'None' are **kept as-is** (no longer converted to 'str').
    - 'Dict[type1, type2]' / 'dict[type1, type2]' / 'object[type1, type2]'
      become a list of {key, value} objects internally, where:
         key.type   = type1
         value.type = type2
      and the final schema does not rely on additionalProperties.
    - If there is no explicit data type and the string looks like only a
      description (e.g. "Python code to generate 5 names"), default the
      field type to 'str'.
    """

    def looks_like_type_snippet(s: str) -> bool:
        """
        Heuristic to decide whether a string looks like a *type* (as opposed to
        a pure description).
        """
        s = s.strip()
        if not s:
            return False

        lower = s.lower()

        # Clearly typed patterns
        if "[" in s and "]" in s:
            # Handles List[str], Dict[str, Any], Enum['A','B'], etc.
            return True

        # Common bare type names
        bare_types = {
            "list", "array", "dict", "object",
            "int", "float", "str", "string",
            "bool", "boolean", "enum", "any", "none",
        }
        if lower in bare_types:
            return True

        # Starts with known generic constructs
        if lower.startswith(("list[", "dict[", "array[", "object[", "enum[")):
            return True

        # Has an explicit "type:" prefix
        if lower.startswith("type:"):
            return True

        return False

    def _normalize_type_string(type_str: str) -> Any:
        """
        Normalize a *type* snippet (no description), e.g.:
          - "list" / "type: list"
          - "dict[str, Any]"
          - "Any"
          - "Enum['A','B']"
        Returns either a (possibly rewritten) type string, or a structured
        schema object (for dicts).
        """
        s = type_str.strip()
        if not s:
            return s

        # Strip optional "type:" prefix so we only work on the core type.
        m = re.search(r'\btype\s*:', s, flags=re.IGNORECASE)
        if m:
            s = s[m.end():].strip()

        lower = s.lower()

        # Normalize pure 'any' to 'str' (but keep 'none' as-is)
        if lower == "any":
            s = "str"
            lower = "str"

        # Replace Any inside composites: List[Any] -> List[str], Dict[str, Any] -> Dict[str, str]
        if re.search(r'\bany\b', s, flags=re.IGNORECASE):
            s = re.sub(r'\bany\b', 'str', s, flags=re.IGNORECASE)
            lower = s.lower()

        # Bare list/array with no inner type → list[str]
        if lower in ("list", "array"):
            return "list[str]"

        # Rewrite dict/object into list of {key, value} objects
        if lower.startswith("dict") or lower.startswith("object"):
            # Default: dict[str, str] if not specified
            key_type_descr = "str"
            value_type_descr = "str"

            if "[" in s and "]" in s:
                inner = s[s.index("[") + 1:s.rindex("]")]
                parts = [p.strip() for p in inner.split(",") if p.strip()]
                if len(parts) == 2:
                    # dict[type1, type2] → key: type1, value: type2
                    key_type_descr = parts[0]
                    value_type_descr = parts[1]

            # Represent dict[K, V] as a list of {key, value} objects.
            return [{"key": key_type_descr, "value": value_type_descr}]

        # Anything else (including Enum[...] and concrete generics) is left as-is.
        return s

    def _rewrite_value_for_openai(field_name: str, value: Any) -> Any:
        """
        Rewrite a single output_format value, preserving description prefixes
        like "some desc, type" where possible.

        If the string looks like *only* a description and not a type snippet,
        default the type to 'str' (e.g. for "Code": "Python code to generate 5 names").
        """
        if isinstance(value, list):
            return [_rewrite_value_for_openai(field_name, v) for v in value]
        if isinstance(value, dict):
            return {k: _rewrite_value_for_openai(k, v) for k, v in value.items()}
        if not isinstance(value, str):
            return value

        raw = value.strip()
        if not raw:
            return value

        # Handle "description, type" pattern (e.g., "some steps, list")
        desc_candidate, possible_type = split_outside_brackets(raw)
        if possible_type is not None:
            normalized = _normalize_type_string(possible_type)
            if isinstance(normalized, str):
                # Reattach description and normalized type
                return f"{desc_candidate}, {normalized}"
            else:
                # Structured schema (e.g., dict → list-of-maps); description
                # can't be attached cleanly, so we return only the structure.
                return normalized

        # No explicit ", type" suffix.
        # If the *entire* string looks like a type, treat it as such;
        # otherwise treat it as pure description → default to 'str'.
        if looks_like_type_snippet(raw):
            return _normalize_type_string(raw)

        # Pure description, no recognizable type → default to 'str'
        return "str"

    def _rewrite_schema_for_openai(s: Any) -> Any:
        if isinstance(s, dict):
            return {k: _rewrite_value_for_openai(k, v) for k, v in s.items()}
        return s

    rewritten = _rewrite_schema_for_openai(schema)
    return convert_schema_to_pydantic(rewritten, main_model=main_model)