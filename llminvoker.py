from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain.memory import ConversationTokenBufferMemory
from langchain.chains import ConversationChain
import keys
from langchain_community.llms import HuggingFaceTextGenInference


def remove_prefix(text, prefix="Ollama: "):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def chat_bot_backend(conversation, model_name, max_context_tokens=2000, max_output_tokens = 2000, assistantPrompt = None, temperatureIn = .5):

    # Define the base template. Notice {systemPrompt} is part of the template.
    base_template = """{systemPrompt}
    Current conversation:
    {{history}}
    Human: {{input}}
    AI:"""
    # If assistantPrompt is provided, replace the placeholder in the base template.
    # Otherwise, use a default prompt or keep an empty string.
    if assistantPrompt:
        system_prompt_content = assistantPrompt
    else:
        system_prompt_content = "This is a conversation with an AI."  # Default prompt, can be adjusted.

    # Now, integrate the actual system prompt content into the template
    template = base_template.format(systemPrompt=system_prompt_content)

    # Initialize the prompt template with the correct variables and the formatted template
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    if model_name.startswith("text-") or model_name.startswith("gpt-"):
        llm = ChatOpenAI(openai_api_key=keys.OPENAI_API_KEY,model_name=model_name, max_tokens=max_output_tokens,temperature=temperatureIn )
    elif model_name.startswith("claude-"):

        llm = ChatAnthropic(model=model_name, anthropic_api_key=keys.ANTHROPIC_API_KEY, max_tokens_to_sample=max_output_tokens, temperature=temperatureIn)
    elif model_name.startswith("gemini-"):
        safetySettings = {
          
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        llm = ChatGoogleGenerativeAI(model=model_name,google_api_key=keys.GOOGLE_API_KEY, 
                                     max_output_tokens=max_output_tokens,  safety_settings=safetySettings,temperature=temperatureIn
                                     )
    elif model_name.startswith("Mixtral"):
        ENDPOINT_URL = 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1'
        llm = HuggingFaceTextGenInference(
            inference_server_url=ENDPOINT_URL,
            max_new_tokens=max_output_tokens,
            top_k=50,
            temperature=0.1,
            repetition_penalty=1.03,
            server_kwargs={
                "headers": {
                    "Authorization": f"Bearer {keys.HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json",
                }
            },
        )
    elif model_name.startswith("Ollama:"):
        inModel = remove_prefix(model_name)
        llm = ChatOllama(model=inModel,temperature=temperatureIn )
        #prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

        # using LangChain Expressive Language chain syntax
        # learn more about the LCEL on
        # /docs/expression_language/why
        #chain = prompt | llm | StrOutputParser()

        # for brevity, response is printed in terminal
        # You can use LangServe to deploy your application for
        # production
        #print(chain.invoke({"topic": "Space travel"}))
    
        

    else:
        raise ValueError(f"Unsupported model: {model_name}")





    # Initialize the conversation memory
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=max_context_tokens)

    # Add the conversation history to the memory
    for user_input, agent_response in conversation[:-1]:
        memory.save_context({"input": user_input}, {"output": agent_response})

    # Get the latest user input
    latest_user_input = conversation[-1][0]

    # Initialize the conversation chain
    conversation_chain = ConversationChain(prompt = PROMPT, llm=llm, memory=memory, verbose = True)

    # Generate the agent's response
    agent_response = conversation_chain.predict(input=latest_user_input)

    return agent_response

if __name__ == "__main__":
    # Define a test conversation
    test_conversation = [
        ["Hello, how can I help you?", "Hi, I need assistance with my account."],
        ["Sure, I can help with that. What is the issue?", "I forgot my password."],
        ["I understand. I can help you reset your password. Do you have access to your email?", ""]
    ]

    # Define the model name and max tokens for this test
    
    test_model_name = "claude-3-opus-20240229"
    
    test_model_name = "gpt-4-turbo-preview"  # 
    test_model_name = "gemini-pro"
    test_model_name = "Mixtral-8x7B-Instruct-v0.1"
    test_model_name = "Ollama: llama3"
    test_model_name = "Ollama: phi"
    test_max_tokens = 2000

    # Run the chat bot backend with the test parameters
    test_response = chat_bot_backend(test_conversation, test_model_name, test_max_tokens, assistantPrompt="System: You are a novel maker")

    # Print out the response from the chat bot
    print("Test response from the chat bot:", test_response)