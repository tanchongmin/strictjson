import heapq
import openai
from openai import OpenAI
import numpy as np
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
    def __init__(self, host = "openai", model = "text-embedding-3-small", database = None):
        '''
        host: Str. The host name for the similarity score service
        model: Str. The name of the model for the host
        database: None / Dict (Key is str for query/key, Value is embedding in List[float])
        Takes in database (dict) of currently generated queries / keys and checks them
        so that you do not need to redo already obtained embeddings
        If you provide a database, the calculated embeddings will be added to this database'''
        self.host = host
        self.model = model
        self.database = database
            
    def __call__(self, query, key) -> float:
        ''' Takes in a query and a key and outputs a similarity score 
        Compulsory:
        query: Str. The query you want to evaluate
        key: Str. The key you want to evaluate'''
     
        database_provided = isinstance(self.database, dict)
        if self.host == "openai":
            client = OpenAI()
            if database_provided and query in self.database:
                query_embedding = self.database[query]
            else:
                query = query.replace("\n", " ")
                query_embedding = client.embeddings.create(input = [query], model=self.model).data[0].embedding
                if database_provided:
                    self.database[query] = query_embedding
                
            if database_provided and key in self.database:
                key_embedding = self.database[query]
            else:
                key = key.replace("\n", " ")
                key_embedding = client.embeddings.create(input = [query], model=self.model).data[0].embedding
                if database_provided:
                    self.database[key] = key_embedding
                
        return np.dot(query_embedding, key_embedding)
        
class Memory:
    ''' Retrieves top k information based on task 
    retriever takes in a query and a key and outputs a similarity score'''
    def __init__(self, memory: list = [], ranker = Ranker()):
        self.memory = memory
        self.ranker = ranker
    def extract(self, task, top_k = 3):
        ''' Performs retrieval of top_k similar memories '''
        memory_score = [self.ranker(memory_chunk, task) for memory_chunk in self.memory]
        top_k_indices = top_k_index(memory_score, top_k)
        return [self.memory[index] for index in top_k_indices]
    def extract_llm(self, task, top_k = 3):
        ''' Performs retrieval via LLMs '''
        res = strict_json(f'You are to output the top {top_k} most similar list items in Memory that meet this description: {task}\nMemory: {self.memory}', '', 
              output_format = {f"top_{top_k}_list": f"Array of top {top_k} most similar list items in Memory, type: list[str]"})
        return res[f'top_{top_k}_list']
    def clear_memory(self):
        ''' Clears all memory '''
        self.memory = []
    
class Agent:
    def __init__(self, agent_name: str = 'Helpful Assistant',
                 agent_description: str = 'A generalist agent meant to help solve problems',
                 memory = Memory(),
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
        - memory. class Memory. Stores the agent's memory for use for retrieval purposes later (to be implemented)
        
        Inputs (optional):
        - **kwargs: Dict. Additional arguments you would like to pass on to the strict_json function
        
        '''
        self.agent_name = agent_name
        self.agent_description = agent_description
        # start with no tasks
        self.task = 'No task assigned'
        self.subtasks_completed = {}
        self.task_completed = False
        self.kwargs = kwargs
        
        # start with default of only llm as the function
        self.function_map = {}
        self.assign_functions([Function(fn_name = 'use_llm', 
                                        fn_description = 'Queries a Large Language Model to perform the task. Used when no other function can do the task', 
                                        output_format = {"Output": "Perform the task and show the output"}),
                              Function(fn_name = 'end_task',
                                       fn_description = 'Ends the current task. Used when task has already been fulfilled',
                                       output_format = {})])
        
    ## Generic Functions ##
    def query(self, query: str, output_format: dict, provide_function_list: bool = False):
        ''' Queries the agent with a query and outputs in output_format. 
        If you want to provide the agent with the context of functions available to it, set provide_function_list to True (default: False)'''
        
        system_prompt = f"You are an agent named {self.agent_name} with the following description: {self.agent_description}."
        if provide_function_list:
            system_prompt += f"\nYou have the following Equipped Functions available:\n{self.list_functions()}"
        system_prompt += query
        
        res = strict_json(system_prompt = system_prompt,
        user_prompt = '',
        output_format = output_format,
        **self.kwargs)
        return res
    
    def status(self):
        ''' Prints prettily the update of the agent's status. 
        If you would want to reference any agent-specific variable, just do so directly without calling this function '''
        print('Agent Name:', self.agent_name)
        print('Agent Description:', self.agent_description)
        print('Available Functions:', list(self.function_map.keys()))
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
                raise Exception('Registered function must be of class Function')

            stored_fn_name = function.__name__
            # if function name is already in use, change name to name + '_1'. E.g. summarise -> summarise_1
            while stored_fn_name in self.function_map:
                stored_fn_name += '_1'

            # add in the function into the function_map
            self.function_map[stored_fn_name] = function
        
    def remove_function(self, function_name: str):
        ''' Removes a function from the agent '''
        if function_name in self.function_map:
            del self.function_map[function_name]
        
    def list_functions(self, as_list: bool = False) -> list:
        ''' Returns the list of functions available to the agent '''
        
        return [f'Name: {name}\n' + str(function) for name, function in self.function_map.items()]
    
    def print_functions(self):
        ''' Prints out the list of functions available to the agent '''
        functions = self.list_functions()
        print('\n'.join(functions))
        
    def select_function(self, task: str = ''):
        ''' Based on the task, output the next function name and input parameters '''
        res = self.query(query = f"Overall Task: {task}\nProcess the Overall Task using exactly one Equipped Function. For Input Values, output empty dictionary {{}} if use_llm or end_task is selected.",
                          output_format = {"Equipped Function": "What Equipped Function to use to solve the task", "Input Values": "Input of function in dictionary form of parameter name: value, type: dict"},
                          provide_function_list = True)
        
        # if model decides to use llm, feed the entire task into the llm
        if res["Equipped Function"] == "use_llm":
            res["Input Values"] = {"task": task}
        
        return res["Equipped Function"], res["Input Values"]
    
    def use_function(self, function_name: str, function_params: dict):
        ''' Uses the function '''
        if function_name == "use_llm":
            res = self.query(query = f'Perform the following task: {function_params["task"]}\nIf asked to generate, generate out fully', 
                            output_format = {"Task Output": "Perform the task concisely and show the output"},
                            provide_function_list = False)
            return res
        elif function_name == "end_task":
            return
        else:
            return self.function_map[function_name](**function_params)
    
    ## Functions for Task Solving ##
    def assign_task(self, task: str):
        ''' Assigns a new task to this agent, and reset subtasks completed list '''
        self.task = task
        self.subtasks_completed = {}
        self.task_completed = False
        
    def get_next_subtask(self):
        ''' Based on what the task is and the subtasks completed, we get the next subtask, function and input parameters'''
        
        res = self.query(query = f"Overall Task: {self.task}\nCompleted Steps: {self.subtasks_completed.items()}\nBased on the overall task and the completed steps, suggest the next step for the Large Language Model to do. Ensure the next step does only what the Task specifies exactly and is a manageable part of the overall task and can be processed by exactly one Equipped Function. If task is completed, call end_task. For Input Values, output empty dictionary {{}} if use_llm or end_task is selected.",
                          output_format = {"Next Step": "Describe what to do for next step", "Equipped Function": "What Equipped Function to use for next step", "Input Values": "Input of function in dictionary form of parameter name: value, type: dict"},
                          provide_function_list = True)
        
        # if the next step is already done before, that will be treated the same way as end_task
        if res["Next Step"] in self.subtasks_completed:
            res["Equipped Function"] = "end_task"
        return res["Next Step"], res["Equipped Function"], res["Input Values"]
    
    def step(self, verbose = True):
        ''' Performs a single step of the task using LLM and available functions '''
        return self.run(task = '', num_steps = 1, verbose = verbose)[-1]
    
    def reply_user(self, verbose = True):
        ''' Generate a reply to the user based on the task and subtasks completed '''
        res = strict_json(system_prompt = f"Respond to the User Task using only the information from the subtasks. Subtasks: {self.subtasks_completed.items()}\n",
                          user_prompt = self.task,
                          output_format = {"Response to User": "Answer the user"},
                          **self.kwargs)
        
        return res["Response to User"]

    def run(self, task: str = '', num_steps: int = 5, verbose = True) -> list:
        ''' Attempts to do the task using LLM and available functions
        Loops through and performs either a function call or LLM call num_steps number of times
        verbose determines whether we print out the agent's thought process when doing the task
        Returns list of outputs for each substep'''
        
        # check if we need to assign a new task
        if task != '':
            self.assign_task(task)
        
        # if task completed, then exit
        if self.task_completed: 
            if verbose:
                print('Task already completed!\n')
                print('Subtasks completed:')
                for key, value in self.subtasks_completed.items():
                    print(f"Subtask: {key}\n{value}\n")
                    
        else:
            # otherwise do the task
            for _ in range(num_steps):           
                # Determine next subtask, or if task is complete
                subtask, function_name, function_params = self.get_next_subtask()
                if function_name == 'end_task':
                    self.task_completed = True
                    if verbose:
                        print('Task completed successfully!\n')
                    break
                    
                if verbose: print('Subtask identified:', subtask)

                # Execute the function for next step
                if function_name == "use_llm":
                    if verbose: 
                        print(f'Getting LLM to perform the following task: {subtask}')
                    res = self.query(query = f'Perform the following task: {subtask}\nOverall context: {self.task}\nCompleted subtasks: {self.subtasks_completed.items()}\nIf asked to generate, generate out fully', 
                                    output_format = {"Task Output": "Perform the task concisely and show the output"},
                                    provide_function_list = False)
                    summary_msg = res["Task Output"]

                else:
                    if verbose: 
                        print(f'Calling function {function_name} with parameters {function_params}')
                    res = self.use_function(function_name, function_params)
                    summary_msg = f'Output of {function_name} with input {function_params}: {res}'

                if verbose: 
                        print(summary_msg, end = '\n\n')
                        
                # add in subtask result into subtasks_completed dictionary
                self.subtasks_completed[subtask] = summary_msg
                

            # check if overall task is complete at the last step if num_steps > 1
            if not self.task_completed:
                subtask, function_name, function_params = self.get_next_subtask()
                if function_name == "end_task":
                    self.task_completed = True
                    if verbose:
                        print('Task completed successfully!\n')
                                  
        return list(self.subtasks_completed.values())
                                  
## TODO ##
class TaskGroup:
    ''' This defines a task group that can be used to solve a task'''
    def __init__(self, task = 'Summarise the news', 
                 agent_pool = [Agent('Summariser', 'Summarises information'), 
                               Agent('Writer', 'Writes news articles'), 
                               Agent('General', 'A generalist agent')],
                conversation_history = [],
                task_progress = [],
                round_robin = False):
        '''
        task: Str. The current task to solve
        agent_pool: List[Agent]. The available agents that we have
        conversation_history: List[Dict]. The past outputs of the agents in JSON Dict.
        task_progress: List[Dict]. The progress of the task, with what has been done at each step stored in a dictionary
        round_robin: Bool (Default: False). Whether or not to call each agent one by one'''
        
        self.task = task
        self.task_progress = task_progress
        self.agent_pool = agent_pool
        self.conversation_history = conversation_history
        self.round_robin = round_robin
        
    def next_subtask(self, task: str, task_progress: list):
        ''' Takes in the task and task progress, and returns the next subtask and agent needed to solve the task
        Returns also whether the task is completed'''
        pass

            
    def step(self):
        ''' Does one step of the task '''
        # 
        pass