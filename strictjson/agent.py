import heapq
import openai
from openai import OpenAI
import numpy as np
import copy
from .base import *

### Helper Functions
def top_k_index(lst, k):
    ''' Given a list lst, find the top k indices corresponding to the top k values '''
    indexed_lst = list(enumerate(lst))
    top_k_values_with_indices = heapq.nlargest(k, indexed_lst, key=lambda x: x[1])
    top_k_indices = [index for index, _ in top_k_values_with_indices]
    return top_k_indices

### Main Classes
class Ranker:
    ''' This defines the ranker which outputs a similarity score given a query and a key'''
    def __init__(self, model = "text-embedding-3-small", ranking_fn = None, database = dict()):
        '''
        model: Str. The name of the model for the host
        ranking_fn: Function. If provided, this will be used to do similarity ranking instead. 
            No database storage possible when using ranking_fn as this function may not generate embeddings
        database: None / Dict (Key is str for query/key, Value is embedding in List[float])
        Takes in database (dict) of currently generated queries / keys and checks them
        so that you do not need to redo already obtained embeddings. 
        New embeddings will also be automatically stored to the database so that you do not have to redo in the future'''
        
        self.model = model
        self.ranking_fn = ranking_fn
        self.database = database
            
    def __call__(self, query, key) -> float:
        ''' Takes in a query and a key and outputs a similarity score 
        Inputs:
        query: Str. The query you want to evaluate
        key: Str. The key you want to evaluate'''
     
        # defaults to OpenAI if ranking_fn is not provided
        if self.ranking_fn is None:
            client = OpenAI()
            if query in self.database:
                query_embedding = self.database[query]
            else:
                query = query.replace("\n", " ")
                query_embedding = client.embeddings.create(input = [query], model=self.model).data[0].embedding
                self.database[query] = query_embedding
                
            if key in self.database:
                key_embedding = self.database[key]
            else:
                key = key.replace("\n", " ")
                key_embedding = client.embeddings.create(input = [key], model=self.model).data[0].embedding
                self.database[key] = key_embedding
                
            return np.dot(query_embedding, key_embedding)
        
        else:
            return self.ranking_fn(query, key)
        
class Memory:
    ''' Retrieves top k memory items based on task 
    - Inputs:
        - `memory`: List. Default: Empty List. The list containing the memory items
        - `top_k`: Int. Default: 3. The number of memory list items to retrieve
        - `mapper`: Function. Maps the memory item to another form for comparison by ranker or LLM. Default: `lambda x: x`
            - Example mapping: `lambda x: x.fn_description` (If x is a Class and the string you want to compare for similarity is the fn_description attribute of that class)
        - `approach`: str. Either `retrieve_by_ranker` or `retrieve_by_llm` to retrieve memory items
            - Ranker is faster and cheaper as it compares via embeddings, but are inferior to LLM-based methods for contextual information
        - `ranker`: `Ranker`. The Ranker which defines a similarity score between a query and a key. Default: OpenAI `text-embedding-3-small` model. 
            - Can be replaced with a function which returns similarity score from 0 to 1 when given a query and key
     '''
    def __init__(self, memory: list = [], top_k: int = 3, mapper = lambda x: x, approach = 'retrieve_by_ranker', ranker = Ranker()):
        self.memory = memory
        self.top_k = top_k
        self.mapper = mapper
        self.approach = approach
        self.ranker = ranker
        
    def retrieve(self, task: str) -> list:
        ''' Performs retrieval of top_k similar memories according to approach stated '''
        if self.approach == 'retrieve_by_ranker':
            return self.retrieve_by_ranker(task)
        else:
            return self.retrieve_by_llm(task)
        
    def retrieve_by_ranker(self, task: str) -> list:
        ''' Performs retrieval of top_k similar memories 
        Returns the memory list items corresponding to top_k matches '''
        memory_score = [self.ranker(self.mapper(memory_chunk), task) for memory_chunk in self.memory]
        top_k_indices = top_k_index(memory_score, self.top_k)
        return [self.memory[index] for index in top_k_indices]
    
    def retrieve_by_llm(self, task: str) -> list:
        ''' Performs retrieval via LLMs 
        Returns the key list as well as the value list '''
        res = strict_json(f'You are to output the top {self.top_k} most similar list items in Memory that meet this description: {task}\nMemory: {[f"{i}. {self.mapper(mem)}" for i, mem in enumerate(self.memory)]}', '', 
              output_format = {f"top_{self.top_k}_list": f"Indices of top {self.top_k} most similar list items in Memory, type: list[int]"})
        top_k_indices = res[f'top_{self.top_k}_list']
        return [self.memory[index] for index in top_k_indices]
    
    def append(self, new_memory):
        ''' Adds a new_memory'''
        self.memory.append(new_memory)
        
    def extend(self, memory_list: list):
        ''' Adds a list of memories '''
        if not isinstance(memory_list, list):
            memory_list = list(memory_list)
        self.memory.extend(memory_list)
        
    def remove(self, memory_to_remove):
        ''' Removes a memory '''
        self.memory.remove(new_memory)
        
    def reset(self):
        ''' Clears all memory '''
        self.memory = []
        
    def isempty(self) -> bool:
        ''' Returns whether or not the memory is empty '''
        return self.memory == []
    
class Agent:
    def __init__(self, agent_name: str = 'Helpful Assistant',
                 agent_description: str = 'A generalist agent meant to help solve problems',
                 max_subtasks: int = 5,
                 memory_bank = {'Function': Memory(top_k = 5, mapper = lambda x: x.fn_name + ': ' + x.fn_description, approach = 'retrieve_by_ranker')},
                 shared_variables = dict(),
                 default_to_llm = True,
                 verbose: bool = True,
                 debug: bool = False,
                 **kwargs):
        ''' 
        Creates an LLM-based agent using description and outputs JSON based on output_format. 
        Agent does not answer in normal free-flow conversation, but outputs only concise task-based answers
        Design Philosophy:
        - Give only enough context needed to solve problem
        - Modularise components for greater accuracy of each component
        
        Inputs:
        - agent_name: String. Name of agent, hinting at what the agent does
        - agent_description: String. Short description of what the agent does
        - max_subtasks: Int. The maximum number of subtasks the agent can have
        - memory_bank: class Dict[Memory]. Stores multiple types of memory for use by the agent. Customise the Memory config within the Memory class.
            - Key: `Function` (Already Implemented Natively) - Does RAG over Task -> Function mapping
            - Can add in more keys that would fit your use case. Retrieves similar items to task / overall plan (if able) for additional context in `get_next_subtasks()` and `use_llm()` function
            - For RAG over Documents, it is best done in a function of the Agent to retrieve more information when needed (so that we do not overload the Agent with information)
        - shared_variables. Dict. Default: empty dict. Stores the variables to be shared amongst inner functions and agents. 
            If not empty, will pass this dictionary by reference down to the inner agents and functions
        - default_to_llm. Bool. Default: True. Whether to default to use_llm function if there is no match to other functions. If False, use_llm will not be given to Agent
        - verbose: Bool. Default: True. Whether to print out intermediate thought processes of the Agent
        - debug: Bool. Default: False. Whether to debug StrictJSON messages
        
        Inputs (optional):
        - **kwargs: Dict. Additional arguments you would like to pass on to the strict_json function
        
        '''
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.max_subtasks = max_subtasks
        self.verbose = verbose
        self.memory_bank = memory_bank
        self.shared_variables = shared_variables
        self.default_to_llm = default_to_llm
        
        self.debug = debug
        
        # reset agent's state
        self.reset()
        
        self.kwargs = kwargs
        
        # start with default of only llm as the function
        self.function_map = {}
        # stores all existing function descriptions - prevent duplicate assignment of functions
        self.fn_description_list = []
        # adds the use llm function
        if self.default_to_llm:
            self.assign_functions([Function(fn_name = 'use_llm', 
                                        fn_description = f'Used only when no other function can do the task', 
                                        output_format = {"Output": "Output of LLM"})])
        # adds the end task function
        self.assign_functions([Function(fn_name = 'end_task',
                                       fn_description = 'Use only when task is completed',
                                       output_format = {})])
        
    ## Generic Functions ##
    def reset(self):
        ''' Resets agent state, including resetting subtasks_completed '''
        self.assign_task('No task assigned')
        self.subtasks_completed = {}
        
    def assign_task(self, task: str, overall_task: str = ''):
        ''' Assigns a new task to this agent. Also treats this as the meta agent now that we have a task assigned '''
        self.task = task
        self.overall_task = task
        # if there is a meta agent's task, add this to overall task
        if overall_task != '':
            self.overall_task = overall_task
            
        self.task_completed = False
        self.overall_plan = None
        
    def query(self, query: str, output_format: dict, provide_function_list: bool = False, task: str = ''):
        ''' Queries the agent with a query and outputs in output_format. 
        If task is provided, we will filter the functions according to the task
        If you want to provide the agent with the context of functions available to it, set provide_function_list to True (default: False)
        If task is given, then we will use it to do RAG over functions'''
        
        # if we have a task to focus on, we can filter the functions (other than use_llm and end_task) by that task
        filtered_fn_list = None
        if task != '':
            filtered_fn_list = self.memory_bank['Function'].retrieve(task)
            # add back use_llm and end_task if present in function_map
            if 'use_llm' in self.function_map:
                filtered_fn_list.append(self.function_map['use_llm'])
            if 'end_task' in self.function_map:
                filtered_fn_list.append(self.function_map['end_task'])
        
        system_prompt = f"You are an agent named {self.agent_name} with the following description: ```{self.agent_description}```\n"
        if provide_function_list:
            system_prompt += f"You have the following Equipped Functions available:\n```{self.list_functions(filtered_fn_list)}```\n"
        system_prompt += query
        
        res = strict_json(system_prompt = system_prompt,
        user_prompt = '',
        output_format = output_format, 
        verbose = self.debug,
        **self.kwargs)
        return res
    
    def status(self):
        ''' Prints prettily the update of the agent's status. 
        If you would want to reference any agent-specific variable, just do so directly without calling this function '''
        print('Agent Name:', self.agent_name)
        print('Agent Description:', self.agent_description)
        print('Available Functions:', list(self.function_map.keys()))
        if len(self.shared_variables) > 0:
            print('Shared Variables:', list(self.shared_variables.keys()))
        print('Task:', self.task)
        if len(self.subtasks_completed) == 0: 
            print("Subtasks Completed: None")
        else:
            print('Subtasks Completed:')
            for key, value in self.subtasks_completed.items():
                print(f"Subtask: {key}\n{value}\n")
        print('Is Task Completed:', self.task_completed)
        
    ## Functions for function calling ##
    def assign_functions(self, function_list: list):
        ''' Assigns a list of functions to be used in function_map '''
        if not isinstance(function_list, list):
            function_list = [function_list]
            
        for function in function_list:
            if not isinstance(function, Function):
                raise Exception('Assigned function must be of class Function')
                
            # Do not assign a function already present
            if function.fn_description in self.fn_description_list:
                continue
            
            stored_fn_name = function.__name__
            # if function name is already in use, change name to name + '_1'. E.g. summarise -> summarise_1
            while stored_fn_name in self.function_map:
                stored_fn_name += '_1'

            # add in the function into the function_map
            self.function_map[stored_fn_name] = function
            
            # add function's description into fn_description_list
            self.fn_description_list.append(function.fn_description)
                                    
            # add function to memory bank if is not the default functions
            if stored_fn_name not in ['use_llm', 'end_task']:
                self.memory_bank['Function'].append(function)
            
        return self
        
    def remove_function(self, function_name: str):
        ''' Removes a function from the agent '''
        if function_name in self.function_map:
            function = self.function_map[function_name]
            # remove actual function from memory bank
            if function_name not in ['use_llm', 'end_task']:
                self.memory_bank['Function'].remove(function)
            # remove function description from fn_description_list
            self.fn_description_list.remove(function.fn_description)
            # remove function from function map
            del self.function_map[function_name]
        
    def list_functions(self, fn_list = None) -> list:
        ''' Returns the list of functions available to the agent. If fn_list is given, restrict the functions to only those in the list '''
        if fn_list is not None and len(fn_list) < len(self.function_map):
            if self.verbose:
                print('Filtered Function Names:', ', '.join([name for name, function in self.function_map.items() if function.fn_name not in ['use_llm', 'end_task'] and function in fn_list]))
            return [f'Name: {name}\n' + str(function) for name, function in self.function_map.items() if function in fn_list]
        else:
            return [f'Name: {name}\n' + str(function) for name, function in self.function_map.items()]                       
    
    def print_functions(self):
        ''' Prints out the list of functions available to the agent '''
        functions = self.list_functions()
        print('\n'.join(functions))
        
    def select_function(self, task: str = ''):
        ''' Based on the task (without any context), output the next function name and input parameters '''
        _, function_name, function_params = self.get_next_subtask(task = task)
            
        return function_name, function_params
    
    def use_agent(self, agent_name: str, agent_task: str):
        ''' Uses an inner agent to do a task for the meta agent. Task outcomes goes directly to subtasks_completed of meta agent '''
        self.use_function(agent_name, {"instruction": agent_task}, agent_task)
        
    def use_function(self, function_name: str, function_params: dict, subtask: str = '', stateful: bool = True):
        ''' Uses the function. stateful means we store the outcome of the function '''
        if function_name == "use_llm":
            if self.verbose: 
                print(f'Getting LLM to perform the following task: {function_params["instruction"]}')
                
             # Add in memory to the LLM
            rag_info = ''
            for name in self.memory_bank.keys():
                # Function is done separately
                if name == 'Function': continue
                # Do not need to add to context if the memory item is empty
                if self.memory_bank[name].isempty(): continue
                rag_info += f'Related {name}: ```{self.memory_bank[name].retrieve(subtask)}```\n'

            res = self.query(query = f'{rag_info}Overall Task:```{self.overall_task}```\nContext:```{self.subtasks_completed}```\nAssigned Subtask: ```{function_params["instruction"]}```\nPerform the Assigned Subtask only - do not just state what has or can be done - actually generate the outcome of Assigned Subtask fully but only within your Agent Capabilities', 
                            output_format = {"Assigned Subtask Outcome": "Generate a full response to the Assigned Subtask"},
                            provide_function_list = False)
            
            res = res["Assigned Subtask Outcome"]
            
            if self.verbose: 
                print('>', res)
                print()
            
        elif function_name == "end_task":
            return
        
        else:
            if self.verbose: 
                print(f'Calling function {function_name} with parameters {function_params}')
                
            res = self.function_map[function_name](**function_params, shared_variables = self.shared_variables)

            # if only one Output key for the json, then omit the output key
            if len(res) == 1 and "Output" in res:
                res = res["Output"]
            
            if self.verbose and res != '': 
                print('>', res)
                print()
                
        if stateful:
            if res == '':
                res = {'Status': 'Completed'}
            
            self.add_subtask_result(subtask, res)

        return res
   
        
    def get_next_subtask(self, task = ''):
        ''' Based on what the task is and the subtasks completed, we get the next subtask, function and input parameters'''
        
        if task == '':
                background_info = f"Assigned Task:```{self.task}```\nAssigned Plan: ```{self.overall_plan}```\nPast Subtasks Completed: ```{self.subtasks_completed}```\n"
        else:
            background_info = f"Assigned Task:```{task}```\n"
                
        # only add overall plan if there is and not evaluasting for single task
        if task == '':
            task = ', '.join(self.overall_plan) if self.overall_plan is not None else task
        task = self.task if task == '' else task
            
        # Add in memory to the Agent
        rag_info = ''
        for name in self.memory_bank.keys():
            # Function RAG is done separately in self.query()
            if name == 'Function': continue
            # Do not need to add to context if the memory item is empty
            if self.memory_bank[name].isempty(): continue
            else:
                rag_info += f'Related {name}: ```{self.memory_bank[name].retrieve(task)}```\n'
                
        # First select the Equipped Function
        res = self.query(query = f'''{background_info}{rag_info}First create an Overall Plan modified from Assigned Plan with an array of steps to do Assigned Task from beginning to end, including uncompleted steps. Then, reflect on Past Subtasks Completed (note not all past subtasks are relevant to Assigned Task) to see what steps in Overall Plan are already completed. Then, generate Overall Plan Completed that outputs True for array elements of Overall Plan that are complete and False otherwsie. Then, generate the Next Step to fulfil the Assigned Task that can be performed by a single Equipped Function. If Assigned Task is completed, output end_task for Equipped Function''',
                         output_format = {"Thoughts": "How to do Assigned Task", "Overall Plan": "Array of steps to complete Assigned Task from beginning to end, type: list", "Reflection": "What has been done and what is still left to do", "Overall Plan Completed": "Whether array elements in Overall Plan are already completed, type: List[bool]", "Next Step": "First non-completed element in Overall Plan", "Equipped Function": f"Name of Equipped Function to use for Next Step, type: Enum{list(self.function_map.keys())}", "Instruction": "Instruction for the Equipped Function if any"},
                          provide_function_list = True,
                         task = task)

        # If equipped function is use_llm, or end_task, we don't need to do another query
        cur_function = self.function_map[res["Equipped Function"]]
        if res["Equipped Function"] == 'use_llm':
            res['Equipped Function Input'] = {'instruction': res['Next Step']}
        elif res['Equipped Function'] == 'end_task':
            res['Equipped Function Input'] = {}
        # Otherwise, if it is only the instruction, no type check needed, so just take the instruction
        elif len(cur_function.variable_names) == 1 and cur_function.variable_names[0].lower() == "instruction":
            res['Equipped Function Input'] = {'instruction': res['Instruction']}
            
        # Otherwise, do another query to get type-checked input parameters and ensure all input fields are present
        else:
            input_format = {}
            fn_description = cur_function.fn_description
            matches = re.findall(r'<(.*?)>', fn_description)
            
            # do up an output format dictionary to use to get LLM to output exactly based on keys and types needed
            for match in matches:
                if ':' in match:
                    first_part, second_part = match.split(':', 1)
                    input_format[first_part] = f'A suitable value, type: {second_part}'
                else:
                    input_format[match] = 'A suitable value'
                    
                    
            res2 = self.query(query = f'''{background_info}{rag_info}Current Subtask: {res["Next Step"]}\nEquipped Function Name: {res["Equipped Function"]}\nEquipped Function Details: {str(cur_function)}\nOutput suitable values for the input parameters of the Equipped Function to fulfil Current Subtask''',
                         output_format = input_format,
                         provide_function_list = False)
            
            res["Equipped Function Input"] = res2

        ## End Task Overrides
        # if the next step is already done before, then end the task. Unless it is in OS mode, then allow it
        if res["Next Step"] in self.subtasks_completed and self.default_to_llm is True:
            res["Equipped Function"] = "end_task"

        # if whole plan is completed, end task
        if False not in res['Overall Plan Completed']:
            res['Equipped Function'] = 'end_task'

        # save overall plan
        if len(res['Overall Plan']) > 0:
            self.overall_plan = res['Overall Plan']
            
        return res["Next Step"], res["Equipped Function"], res["Equipped Function Input"]
        
    def add_subtask_result(self, subtask, result):
        ''' Adds the subtask and result to subtasks_completed
        Keep adding (num) to subtask str if there is duplicate '''
        subtask_str = str(subtask)
        count = 2
        
        # keep adding count until we have a unique id
        while subtask_str in self.subtasks_completed:
            subtask_str = str(subtask) + f'({count})'
            count += 1
            
        self.subtasks_completed[subtask_str] = result
        
    def remove_last_subtask(self):
        ''' Removes last subtask in subtask completed. Useful if you want to retrace a step '''
        if len(self.subtasks_completed) > 0:
            removed_item = self.subtasks_completed.popitem()
        if self.verbose:
            print(f'Removed last subtask from subtasks_completed: {removed_item}')
        
    def summarise_subtasks_completed(self, task: str = ''):
        ''' Summarise the subtasks_completed list according to task '''

        output = self.reply_user(task)
        # Create a new summarised subtasks completed list
        self.subtasks_completed = {f"Summary of {task}": output}
        
    def reply_user(self, query: str = '', stateful: bool = True):
        ''' Generate a reply to the user based on the query / agent task and subtasks completed
        If stateful, also store this interaction into the subtasks_completed'''
        
        my_query = self.task if query == '' else query
            
        res = self.query(query = f'Context: ```{self.subtasks_completed}```\nAssigned Task: ```{my_query}```\nGenerate a response to the Assigned Task using the Context only', 
                                    output_format = {"Response to Assigned Task": "Use the Context as ground truth to respond to as many parts of the Assigned Task as possible. Do not respond with any information not from Context."},
                                    provide_function_list = False)
        
        res = res["Response to Assigned Task"]
        
        if self.verbose:
            print(res)
        
        if stateful:
            self.add_subtask_result(my_query, res)
        
        return res

    def run(self, task: str = '', overall_task: str = '', num_subtasks: int = 0) -> list:
        ''' Attempts to do the task using LLM and available functions
        Loops through and performs either a function call or LLM call up to max_steps number of times
        If overall_task is filled, then we store it to pass to the inner agents for more context'''
            
        # Assign the task
        if task != '':
            self.task_completed = False
            # If meta agent's task is here as well, assign it too
            if overall_task != '':
                self.assign_task(task, overall_task)
            else:
                self.assign_task(task)
            
        # check if we need to override num_steps
        if num_subtasks == 0:
            num_subtasks = self.max_subtasks
        
        # if task completed, then exit
        if self.task_completed: 
            if self.verbose:
                print('Task already completed!\n')
                print('Subtasks completed:')
                for key, value in self.subtasks_completed.items():
                    print(f"Subtask: {key}\n{value}\n")
                    
        else:
            # otherwise do the task
            for i in range(num_subtasks):           
                # Determine next subtask, or if task is complete. Always execute if it is the first subtask
                subtask, function_name, function_params = self.get_next_subtask()
                if function_name == 'end_task':
                    self.task_completed = True
                    if self.verbose:
                        print('Task completed successfully!\n')
                    break
                    
                if self.verbose: print('Subtask identified:', subtask)

                # Execute the function for next step
                res = self.use_function(function_name, function_params, subtask)
                         
            # check if overall task is complete at the last step if num_steps > 1
            if not self.task_completed and num_subtasks > 1:
                subtask, function_name, function_params = self.get_next_subtask()
                if function_name == "end_task":
                    self.task_completed = True
                    if self.verbose:
                        print('Task completed successfully!\n')

        return list(self.subtasks_completed.values())
    
    ## This is for Multi-Agent uses
    def to_function(self, meta_agent):
        ''' Converts the agent to a function so that it can be called by another agent
        The agent will take in an instruction, and output the result after processing'''

        # makes the agent appear as a function that takes in an instruction and outputs the executed instruction
        my_fn = Function(fn_name = self.agent_name,
                             fn_description = f'Agent Description: ```{self.agent_description}```\nExecutes the given <instruction>',
                             output_format = {"Output": "Output of instruction"},
                             external_fn = Agent_External_Function(self, meta_agent))
        
        return my_fn
    
    def assign_agents(self, agent_list: list):
        ''' Assigns a list of Agents to the main agent, passing in the meta agent as well '''
        if not isinstance(agent_list, list):
            agent_list = [agent_list]
        self.assign_functions([agent.to_function(self) for agent in agent_list])
        return self
    
    ## Function aliaises
    list_function = list_functions
    assign_agent = assign_agents
    print_function = print_functions
    assign_function = assign_functions
    
class Agent_External_Function:
    ''' Creates a Function-based version of the agent '''
    def __init__(self, agent: Agent, meta_agent: Agent):
        ''' Retains the instance of the agent as an internal variable '''
        self.agent = agent
        self.meta_agent = meta_agent

    def __call__(self, instruction: str):
        ''' Calls the inner agent to perform an instruction. The outcome of the agent goes directly into subtasks_completed
        No need for return values '''
        # make a deep copy so we do not affect the original agent
        if self.agent.verbose:
            print(f'\n### Start of Inner Agent: {self.agent.agent_name} ###')
        agent_copy = copy.deepcopy(self.agent)
        
        # take the shared variables from the meta agent
        agent_copy.shared_variables = self.meta_agent.shared_variables
        
        # provide the subtasks completed and debug capabilities to the inner agents too
        agent_copy.debug = self.meta_agent.debug
        agent_copy.subtasks_completed = self.meta_agent.subtasks_completed

        output = agent_copy.run(instruction, self.meta_agent.overall_task)
        if self.agent.verbose:
            print(f'### End of Inner Agent: {self.agent.agent_name} ###\n')
