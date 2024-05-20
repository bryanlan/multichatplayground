import gradio as gr
from modelindex import model_specs
from classChats import TextBox, ChatBot


with gr.Blocks() as demo:
    with gr.Row():        
        with gr.TabItem("Workspace"):
            with gr.Row():
                with gr.Column():
                    filtered_models = {
                        model_info['friendly_name']: "n/a tokens/second" 
                        for model_name, model_info in model_specs.items() 
                        if model_info['is_used'] and model_info['is_local']}
                    formatted_labels = [f"{key}: {value}" for key, value in filtered_models.items()]
                    formatted_label_str = "\n".join(formatted_labels)
                    
                    perfLabel = gr.Textbox(
                        label="Performance",
                        value=formatted_label_str,
                        interactive=False
                    )
                with gr.Column(scale=4):
                    activeThread = gr.Textbox(label="Active Chat", value="default")
                    model_choice = gr.Dropdown(list(model_specs.keys()), label="Select Model", multiselect=True, value= next(iter(model_specs)))
                    temp_slider = gr.Slider(maximum=1, minimum=0, step=.1, value=.7)
                    chatbot = gr.Chatbot(label="Work Area")
                    msg = gr.Textbox(label="Prompt", lines=10, max_lines=50, placeholder="Enter prompt here")
                    clear_chat = gr.ClearButton([msg, chatbot])
                    save_button = gr.Button("Save")
                    remove_button = gr.Button("Remove")
                    thread_choice = gr.Dropdown(choices=["default", "thread 1", "thread 2"], label="Load chat")

    
        model_chatbots = []

        for model_name, model_info in model_specs.items():
            if model_info['is_used']:
                with gr.TabItem(model_info['friendly_name']):
                    with gr.Row():
                        model_chatbot = gr.Chatbot(label=f"{model_info['friendly_name']} Conversation")
                        model_chatbots.append(model_chatbot)

    # Event handlers

    ThreadText = ChatBot(full_path = "savedChats.json", json_object = {},full_json_object = {}, name_text_box=activeThread,
                         dropdown = thread_choice, clear_button= clear_chat, save_button = save_button, remove_button = remove_button,
                         primary_chatbot=chatbot,secondary_chatbots=model_chatbots, model_specs = model_specs, model_choice=model_choice, 
                         temp_slider=temp_slider,perf_label=perfLabel, msg = msg,sys_msg_text= "System: You are a helpful assistant")
    ThreadText.bind_event_handlers()
    ThreadText.initialize()

if __name__ == "__main__":
    demo.launch()
