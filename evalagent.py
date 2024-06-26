from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser, create_tool_calling_agent, tool, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain.memory import ConversationTokenBufferMemory, ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, BaseMessage

from llminvoker import execute_llms, clean_chat_history
import keys
from typing import Any, Dict, List
import asyncio
import nest_asyncio
import copy

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()
class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print(f"CallbackPrompt: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print(f"CallbackResponse: {response.generations[0][0].text}")
        print("---")

custom_handler = CustomHandler()

tool_used = False
llm_results = None
formatted_label_str = ""
modelInfo = {}

def concatenate_dict_pairs(d, result=None):
    if result is None:
        result = []
    
    if isinstance(d, str):
        result.append(d)
        result.append('\n')
    elif isinstance(d, dict):
        for key, value in d.items():
            if isinstance(value, dict):
                result.append(f"{key}:")
                concatenate_dict_pairs(value, result)
            else:
                result.append(f"{key}:{value}")
            result.append('\n')  # Add a newline after each value

    return ''.join(result).strip()  # Join and strip the trailing newline

async def grade_multiple_llms_response_single_async(prompt: str, evalCriteria: str) -> str:
    global tool_used, modelInfo, llm_results

    formatted_chatHistory = [[(prompt,"")] for model_name in modelInfo['model_names']]
   
    # execute the question on all LLMs
    llm_results, formatted_label_str = await execute_llms(
                modelInfo['model_names'], modelInfo['model_specs'], modelInfo['temperature'], "You are a helpful assistant", formatted_chatHistory)
    
    if evalCriteria:
        evalPrompt =  evalCriteria
    else:
        evalPrompt = "Evaluate these multiple LLM responses on a scale of 1 to 100, explaining your reasoning. "

    formatted_chatHistory = [[(evalPrompt + concatenate_dict_pairs(llm_results),)]]
    llm_eval, formatted_label_str = await execute_llms(
                [modelInfo['model_names'][0]], modelInfo['model_specs'], modelInfo['temperature'], "You evaluate LLM response quality.", formatted_chatHistory)


    tool_used = True
    return llm_eval


@tool
def grade_multiple_llms_response_multiple(prompt: str, numberEval: int, evalCriteria: str) -> str:
    
    '''ONLY use when asked to evaluate multiple LLMs reponses to a question multiple times (must be a specific number >1 ) by nested looping through the responses and grading them. Inputs: 1) prompt: str - the question to evaluate 
    2) numberEval: int - the number of times to ask the question to multiple LLMs, cannot be 1 3) evalCritera: optional str the criteria defined for the evaluation, if specified. 
    Outputs: 1) String containing the scores for each iteration of asking the question'''
    global tool_used
    tool_used = True
    print("Entering grade response multiple, Prompt:" + prompt+ ". criteria:" + evalCriteria + ". Iterations:"+str(numberEval))
    if evalCriteria:
        evalPrompt =  evalCriteria
    else:
        evalPrompt = "Evaluate these multiple LLM responses on a scale of 1 to 100, explaining your reasoning. "
    llmResponses = ""
    for i in range(numberEval - 1):
        loop = asyncio.get_event_loop()
        response=loop.run_until_complete(grade_multiple_llms_response_single_async(prompt, evalPrompt))
        if isinstance(response, dict):
            response = concatenate_dict_pairs(response)
        llmResponses+=response
    

    return llmResponses

@tool
def grade_multiple_llms_response_single(prompt: str, evalCriteria: str) -> dict:
    '''Evaluates multiple LLM responses to a question by getting the LLM responses. Inputs: 1) prompt: str - the question and responses from the multiple LLMs 
    evalCritera: optional str the criteria defined for the evaluation, if specified. 
    Outputs: 1) evalInfo: dict - keys are llm model names, value are responses which AI will then grade according to evalCriteria input parameter'''
    print("Entering grade response single, Prompt:" + prompt+ ". Criteria:" + evalCriteria)
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(grade_multiple_llms_response_single_async(prompt, evalCriteria))


class LangChainAgent:
    def __init__(self, chat_history):
        MAX_AGENT_TOKENS = 4000
        self.custom_handler = CustomHandler()
        self.tools = [
            grade_multiple_llms_response_multiple,
            grade_multiple_llms_response_single
        ]
        self.chat_history = chat_history

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
         # Initialize the conversation memory
        self.memory = ConversationBufferMemory(max_token_limit=MAX_AGENT_TOKENS, memory_key="chat_history", return_messages=True) # llm=self.chat, 
        #self.chat_history = [("What is your name?", "My name is Mr. AI"),("placeholder",)]
        # Add the conversation history to the memory
        for user_input, agent_response in self.chat_history[:-1]:
            self.memory.save_context({"input": user_input}, {"output": agent_response})

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            memory = self.memory,max_iterations=5, 
            max_execution_time=1000
       
        )
       
    def invoke(self, user_input: str) -> str:
        chat_history = self.memory.buffer_as_messages
        assert isinstance(chat_history, list), "chat_history is not a list"
        assert all(isinstance(msg, (HumanMessage, AIMessage)) for msg in chat_history), "chat_history contains invalid message types"
 
        result = self.agent_executor.invoke({"input": user_input, "chat_history":chat_history})
        return result
    
   

    async def run_agent_and_llms(self, model_names, model_specs, temperature, sys_msg, cleaned_chat_histories, user_input):
        global tool_used, llm_results, formatted_label_str, modelInfo
        first_model = model_names[0]
        modelInfo = {'model_names':model_names, 'model_specs':model_specs, 'temperature': temperature, 'sys_msg':sys_msg}
        if model_specs[first_model]["is_local"]:
            # If the first model is local, run execute_llms directly and return the results
            results, formatted_label_str = await execute_llms(
                model_names, model_specs, temperature, sys_msg, cleaned_chat_histories
            )
            return results, formatted_label_str
        tool_used = False

        agent_result = self.invoke(user_input)
        
        if tool_used:
            # pad the results so that the actual responses will show up for their respective LLMs, and the eval will show for the main LLM.
            results_simple = {model_name: llm_results[model_name] +  ("\n\n Evaluations of Multiple LLMs: \n\n" + concatenate_dict_pairs(agent_result['output']) if model_name == first_model else '') for model_name in model_names}
            return results_simple, ""

        # If the agent did not use any tools, run execute_llms
        llm_results, formatted_label_str = await execute_llms(
            model_names, model_specs, temperature, sys_msg, cleaned_chat_histories
        )
        return llm_results, formatted_label_str
        
if __name__ == "__main__":
    agent = LangChainAgent()
    user_input = "Evaluate the different llms to see how they handle this question: How is a duck like quantum mechanics?"
    result = agent.invoke(user_input)
    print(result)
