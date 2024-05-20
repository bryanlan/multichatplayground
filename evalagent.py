from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_tool_calling_agent, tool, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from llminvoker import execute_llms, clean_chat_history
import keys
from typing import Any, Dict, List
import concurrent.futures
import threading
import asyncio
import copy

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print(f"CallbackPrompt: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"CallbackResponse: {response.generations[0][0].text}")
        print("---")

custom_handler = CustomHandler()
# have to define some tools and globals outside of the class because of langchain limitations
tool_used = False
llm_thread = None
llm_stop_event = threading.Event()
thread_completed = threading.Event()
llm_results = None
formatted_label_str = ""
modelInfo = {}

async def grade_multiple_llms_response_single_async(prompt: str, evalCriteria: str) -> dict:
    global tool_used, modelInfo, llm_results
    #if evalCriteria:
    #    eCriteria = evalCriteria
    #else:
    #    eCriteria = " a scale of 1 to 10."
    formatted_chatHistory = [[(prompt,"")] for model_name in modelInfo['model_names']]

    cancel_llm_futures()
   
    # execute the question on all LLMs
    llm_results, formatted_label_str = await execute_llms(
                modelInfo['model_names'], modelInfo['model_specs'], modelInfo['temperature'], "You are a helpful assistant", formatted_chatHistory)
    llm_response = copy.deepcopy(llm_results)
    #llm_response['Next Step'] = "Now AI should evaluate previous responses according to " + eCriteria
    tool_used = True
    return llm_response

@tool
def grade_multiple_llms_response_multiple(prompt: str, numberEval: int, evalCriteria: str) -> dict:
    
    '''ONLY use when asked to evaluate multiple LLMs reponses to a question multiple times (must be a specific number >1 ) by nested looping through the responses and grading them. Inputs: 1) prompt: str - the question to evaluate 
    2) numberEval: int - the number of times to ask the question to multiple LLMs, cannot be 1 3) evalCritera: optional str the criteria defined for the evaluation, if specified. 
    Outputs: 1) Dictionary containing keys for each LLM with each key having a value consisting of a list of evaluation scores'''
    cancel_llm_futures()
    global tool_used
    tool_used = True
    llmResponses = []
    for i in range(numberEval - 1):
        llmResponses.append(grade_multiple_llms_response_single(prompt))
    evalInfo = {'LLM_Claude': [6, 7, 8, 9], 'LLM_Ollamallama3': [7, 7, 7, 7], 'LLM_OllamaPhi3': [4, 5, 6, 7], 'LLM_OnnxDMLPhi3': [5, 6, 7, 8]}
    return evalInfo

@tool
def grade_multiple_llms_response_single(prompt: str, evalCriteria: str) -> dict:
    '''Evaluates multiple LLM responses to a question by getting the LLM responses. Inputs: 1) prompt: str - the question and responses from the multiple LLMs 
    evalCritera: optional str the criteria defined for the evaluation, if specified. 
    Outputs: 1) evalInfo: dict - keys are llm model names, value are responses which AI will then grade according to evalCriteria input parameter'''

    # wrap the async function that does the work
    return asyncio.run(grade_multiple_llms_response_single_async(prompt, evalCriteria))





def cancel_llm_futures():
    global llm_stop_event
    global llm_thread

    llm_stop_event.set()
    if llm_thread:
       llm_thread.join()


class LangChainAgent:
    def __init__(self):
        self.custom_handler = CustomHandler()
        self.tools = [
            #get_multiple_llms_response_single,
            grade_multiple_llms_response_multiple,
            grade_multiple_llms_response_single
        ]

        self.system =  '''Respond to the human as helpfully and accurately as possible. You have access to the following tools:
            {tools}
            Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

            Valid "action" values: "Final Answer" or {tool_names}

            Provide only ONE action per $JSON_BLOB, as shown:

            ```
            {{
            "action": $TOOL_NAME,
            "action_input": $INPUT
            }}
            ```

            Follow this format:

            Question: input question to answer
            Thought: consider previous and subsequent steps
            Action:
            ```
            $JSON_BLOB
            ```
            Observation: action result
            ... (repeat Thought/Action/Observation N times)
            Thought: I know what to respond
            Action:
            ```
            {{
            "action": "Final Answer",
            "action_input": "Final response to human"
            }}

            Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools but only if absolutely required. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'''

        self.human = '''{input}

            {agent_scratchpad}

            (reminder to respond in a JSON blob no matter what)'''
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", self.human),
        ])
        self.chat = ChatOpenAI(
            temperature=0,
            openai_api_key=keys.OPENAI_API_KEY,
            model='gpt-4o',
            callbacks=[self.custom_handler]
        )
        self.agent = create_structured_chat_agent(self.chat, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            return_intermediate_steps=False,
        )
       
    def invoke(self, user_input: str) -> str:
        result = self.agent_executor.invoke({"input": user_input})
        return result
    
    
    async def execute_llms_wrapper(self, model_names, model_specs, temperature, sys_msg, cleaned_chat_histories):
        global llm_stop_event, thread_completed, llm_results, formatted_label_str

        llm_stop_event.clear()
        thread_completed.clear()
        
        llm_results, formatted_label_str = await execute_llms(
            model_names, model_specs, temperature, sys_msg, cleaned_chat_histories
        )
        thread_completed.set()



    def start_execute_llms(self, model_names, model_specs, temperature, sys_msg, cleaned_chat_histories):
        global llm_thread
        llm_thread = threading.Thread(
            target=asyncio.run,  # Use asyncio.run to run the coroutine in a new thread
            args=(self.execute_llms_wrapper(model_names, model_specs, temperature, sys_msg, cleaned_chat_histories),)
        )
        llm_thread.start()

    def concatenate_dict_pairs(self, d, result=None):
        if result is None:
            result = []
        
        if isinstance(d, str):
            result.append(d)
            result.append('\n')
        elif isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict):
                    result.append(f"{key}:")
                    self.concatenate_dict_pairs(value, result)
                else:
                    result.append(f"{key}:{value}")
                result.append('\n')  # Add a newline after each value
    
        return ''.join(result).strip()  # Join and strip the trailing newline

    async def run_agent_and_llms(self, model_names, model_specs, temperature, sys_msg, cleaned_chat_histories, user_input):
        global tool_used, llm_thread, llm_results, formatted_label_str, thread_completed, modelInfo
        first_model = model_names[0]
        # assign all revelant model info to the global so that the langchain tools can use it (since they won't work under a class defn)
        modelInfo = {'model_names':model_names, 'model_specs':model_specs, 'temperature': temperature, 'sys_msg':sys_msg}
        if model_specs[first_model]["is_local"]:
            # If the first model is local, run execute_llms directly and return the results
            results, formatted_label_str = await execute_llms(
                model_names, model_specs, temperature, sys_msg, cleaned_chat_histories
            )
            return results, formatted_label_str


        # If the first model is not local, run the agent and execute_llms in parallel
        
        tool_used = False

        with concurrent.futures.ThreadPoolExecutor() as executor:
            agent_future = executor.submit(self.invoke, user_input)
            self.start_execute_llms(
                model_names, model_specs, temperature, sys_msg, cleaned_chat_histories
            )

            agent_result = agent_future.result()
            if tool_used:
                if thread_completed.is_set():
                    print("Thread of initial llm execution completed normally.")
                else:
                    print("Thread of initial llm execution was preempted and halted by tools invocation.")
                # pad the results so that the actual responses will show up for their respective LLMs, and the eval will show for the main LLM.
                results_simple = {model_name: llm_results[model_name] +  ("\n\n Evaluations of Multiple LLMs: \n\n" + self.concatenate_dict_pairs(agent_result['output']) if model_name == first_model else '') for model_name in model_names}

                return results_simple, ""

            if llm_thread:
                llm_thread.join()          
            if thread_completed.is_set():
                print("Thread of initial llm execution completed normally.")
            else:
                print("Thread of initial llm execution was preempted and halted by tools invocation.")
      
        return llm_results, formatted_label_str

        
    
if __name__ == "__main__":
    agent = LangChainAgent()
    user_input = "Evaluate the different llms to see how they handle this question: How is a duck like quantum mechanics?"
    #user_input = "How is a duck like quantum mechanics?"
    result = agent.invoke(user_input)
    print(result)
