import os
import json

from llminvoker import clean_chat_history  
from evalagent import LangChainAgent
import gradio as gr
import asyncio

class BaseBox:
    def __init__(self, full_path,  dropdown, clear_button, save_button, remove_button, json_object, full_json_object, name_text_box):
        # json object is a dictionary, possibly nested, where keys Threads, Phases, ActivePhase, and ActiveThread can be added
        # full json object is the full file that needs to be written.
        self.full_path = full_path       
        self.dropdown = dropdown
        self.clear_button = clear_button
        self.save_button = save_button
        self.remove_button = remove_button
        
        self.json_object = json_object
        self.full_json_object = full_json_object
        self.name_text_box = name_text_box
       
       

       
    
    def initialize(self, full_path=None,  dropdown=None, clear_button=None, save_button=None, remove_button=None, json_object=None, full_json_object = None, name_text_box = None):
        # update if we have an updated value that isn't none
        self.full_path = full_path or self.full_path         
        self.dropdown = dropdown or self.dropdown
        self.clear_button = clear_button or self.clear_button
        self.save_button = save_button or self.save_button
        self.remove_button = remove_button or self.remove_button
        self.full_json_object = full_json_object or self.full_json_object
        
        self.json_object = json_object or self.json_object
        self.name_text_box = name_text_box or self.name_text_box
   
       

    
class TextBox(BaseBox):
    def __init__(self,  textbox, *args, **kwargs):

        super().__init__( *args, **kwargs)
       
        self.text_box = textbox
        self.bind_event_handlers()
    
    def initialize(self, full_path=None, dropdown=None, clear_button=None, save_button=None, remove_button=None, json_object=None, full_json_object = None, name_text_box=None, textbox=None):
        super().initialize(full_path, dropdown, clear_button, save_button, remove_button,  json_object, full_json_object, name_text_box)
        
        
        self.text_box = textbox or self.text_box
 
        if "ActivePhase" not in json_object:
            json_object['ActivePhase'] = 'default'

    def bind_event_handlers(self):
        self.clear_button.click(self._clear_text, [], [self.text_box])
        self.save_button.click(self.save_text, [self.text_box, self.name_text_box], [self.text_box, self.dropdown])
        self.remove_button.click(self._remove_text, [self.name_text_box], [self.text_box, self.dropdown, self.name_text_box])
        # phase_choice.change(change_active_phase, inputs=[phase_choice], outputs=[activeThread, activePhase, chatbot, phase_text])- chatbot and active thread not necessary
        self.dropdown.change(self.load_text, [self.dropdown], [self.text_box, self.name_text_box])


 

    def _update_json_object(self, new_json_object):
        self.json_object = new_json_object
        if "ActivePhase" not in self.json_object:
            self.json_object['ActivePhase'] = 'default'
        #self.load_text(self.json_object['ActivePhase'])

    def get_dropdown_choices(self):
        return list(self.json_object['Phases'].keys())
    
    def _set_dropdown_choices(self):
        self.dropdown.choices = self.get_dropdown_choices()
        self.dropdown.update(choices=self.dropdown.choices)
        
    def get_text(self):
        return self.json_object['Phases'][self.json_object['ActivePhase']]
       
    def set_text(self, text):
        self.text_box.value = text

    def _clear_text(self):
        return ""
    
    def save_text(self, text, name): 
        # Directly manipulate the provided JSON object
        self.json_object['Phases'][name] = text

        # Save the updated JSON object back to the file
        with open(self.full_path, "w") as file:
            json.dump(self.full_json_object, file, indent=4)
        choices_update = gr.update(choices=self.get_dropdown_choices(), value=name)
        # Update dropdown choices
        return text, choices_update

    def _remove_text(self, name):
        # Remove the entry from the JSON object but don't delete default
        if name != "default":
            if name in self.json_object['Phases']:
                del self.json_object['Phases'][name]
                self.json_object['ActivePhase'] = "default"

                # Save the updated JSON object back to the file
                with open(self.full_path, "w") as file:
                    json.dump(self.full_json_object, file, indent=4)
            text = self.json_object['Phases'].get("default", "")
            choices_update = gr.update(choices=self.get_dropdown_choices())
    
            return [text, choices_update, "default"]



    def load_text(self, name):
        # Load the text from the JSON object
        text = self.json_object['Phases'].get(name, "")
        self.json_object['ActivePhase'] = name
        
        return text, name

class ChatBot(BaseBox):
    def __init__(self,  primary_chatbot, secondary_chatbots, model_specs, model_choice, temp_slider,perf_label, msg, sys_msg_text, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.primary_chatbot = primary_chatbot
        self.secondary_chatbots = secondary_chatbots
        self.model_specs = model_specs
        self.model_choice = model_choice
        self.temp_slider = temp_slider
        self.sys_msg_text = sys_msg_text
        self.message_selected = gr.State(None)
        self.msg = msg
        self.perf_label = perf_label

    
       
    def initialize(self, full_path=None, dropdown=None, clear_button=None, save_button=None, remove_button=None, json_object=None, full_json_object = None, name_text_box = None,
                   primary_chatbot = None, secondary_chatbots = None, model_specs = None, model_choice = None, temp_slider = None, perf_label  = None, msg = None):
        super().initialize(full_path, dropdown, clear_button, save_button, remove_button,  json_object, full_json_object, name_text_box)
        
        self.primary_chatbot = primary_chatbot or self.primary_chatbot
        self.secondary_chatbots = secondary_chatbots or self.secondary_chatbots
        self.model_specs = model_specs or self.model_specs
        self.model_choice = model_choice or self.model_choice
        self.temp_slider = temp_slider or self.temp_slider
        self.msg = msg or self.msg
        self.perf_label = perf_label or self.perf_label
 
        if "ActiveThread" not in self.json_object:
            self.json_object['ActiveThread'] = 'default'

    def bind_event_handlers(self):
        
        self.clear_button.click(self._clear_text, [*self.secondary_chatbots], [self.primary_chatbot, *self.secondary_chatbots])
        self.save_button.click(self._save_text, [self.primary_chatbot, self.name_text_box, *self.secondary_chatbots], [self.name_text_box, self.dropdown])
        self.remove_button.click(self._remove_text, [self.name_text_box], [self.dropdown,self.name_text_box, self.primary_chatbot, *self.secondary_chatbots])

        self.dropdown.change(self.load_text, [self.dropdown], [self.name_text_box, self.primary_chatbot, *self.secondary_chatbots])
        self.msg.submit(self._respond,[self.msg, self.primary_chatbot, self.message_selected, self.model_choice, self.temp_slider, self.perf_label,*self.secondary_chatbots],
                        [self.msg, self.primary_chatbot, self.message_selected,self.perf_label, *self.secondary_chatbots])

        self.primary_chatbot.select(self._save_selected_message, None, [self.msg, self.message_selected])
  

    def _update_json_object(self, new_json_object):
        self.json_object = new_json_object
        if "ActiveThread" not in self.json_object:
            self.json_object['ActiveThread'] = 'default'
        #self.load_text(self.json_object['ActivePhase'])

    def get_dropdown_choices(self):
        return list(self.json_object['Threads'].keys())
    
    def _set_dropdown_choices(self):
        self.dropdown.choices = self.get_dropdown_choices()
        self.dropdown.update(choices=self.dropdown.choices)
        
    def get_text(self):
        return self.json_object['Threads'][self.json_object['ActiveThread']]
    
    def get_ActiveThread(self):
        return self.json_object['ActiveThread']
           
    def update_system_message(self, newSysMsg):
        self.sys_msg_text = newSysMsg

    def _clear_text(self, *model_chatbots):
        empty_histories = [[] for _ in range(len(model_chatbots) +1)]
        return empty_histories
    
    def _save_text(self, chat_history, thread_name, *model_chatbots): 
 

        chat_histories = [chat_history] + list(model_chatbots)
        model_names = []
        # Populate model_names based on the order of models defined in model_specs
        for key in self.model_specs:
            friendly_name = self.model_specs[key]['friendly_name']
            model_names.append(friendly_name)

        if 'Threads' not in self.json_object:
            self.json_object['Threads']={}
            self.json_object['Threads']['default']={}
        
        if thread_name not in self.json_object['Threads']:
            self.json_object['Threads'][thread_name] = {}
        # Update the chat history for each model
        for model_name, chat_history in zip(model_names, chat_histories):
            self.json_object['Threads'][thread_name][model_name] = chat_history
        
     
        # Save the updated JSON object back to the file
        with open(self.full_path, "w") as file:
            json.dump(self.full_json_object, file, indent=4)
        choices_update = gr.update(choices=self.get_dropdown_choices(), value=thread_name)
        # Update dropdown choices
        return thread_name, choices_update
    

    def _remove_text(self, name):
        # Remove the entry from the JSON object but don't delete default
        if name != "default":
            if name in self.json_object['Threads']:
                del self.json_object['Threads'][name]
                self.json_object['ActiveThread'] = "default"

                # Save the updated JSON object back to the file
                with open(self.full_path, "w") as file:
                    json.dump(self.full_json_object, file, indent=4)
            selected_name,*chat_text = self.load_text("default")
            #text = self.json_object['Threads'].get("default", "")
            choices_update = gr.update(choices=self.get_dropdown_choices())
    
            return [choices_update, selected_name, *chat_text]
  


    # call load text with default thread which gets set during initialization 
    def load_text(self, name):
        # Load the text from the JSON object
        text = self.json_object['Threads'].get(name, "")
        self.json_object['ActiveThread'] = name
        # Initialize empty lists for ordered chat histories
        ordered_chat_histories = []
        # Extract raw chat histories for the active thread, which now contains model-specific chats
        raw_chat_histories = self.json_object['Threads'].get(name, {})
        # Extract and order the chat histories based on model_specs alignment
        for key in self.model_specs:
            friendly_name = self.model_specs[key]['friendly_name']
            # Find which model_name the friendly_name corresponds to
            for model_name, chat_history in raw_chat_histories.items():
                if model_name == friendly_name:
                    ordered_chat_histories.append(chat_history)
                    break
            else:  # If the model wasn't found, append an empty history
                ordered_chat_histories.append([])
      
     
        # Return the ordered chat histories along with other thread information
        return name,  *ordered_chat_histories
        
    
    
    def _save_selected_message(self, msg: gr.SelectData):
        return msg.value, msg.value

    
    

   
 

    def _respond(self, message, chat_history, message_selected, model_names, temp_slider, perf_label, *model_chatbots):
        if message_selected is None:
            chat_history.append((message, ""))
            for model_chatbot in model_chatbots:
                model_chatbot.append((message, ""))

            new_chat_history = clean_chat_history(chat_history, self.model_specs)
            new_model_chatbots = [clean_chat_history(model_chatbot, self.model_specs) for model_chatbot in model_chatbots]

            if not model_names:
                model_names = [list(self.model_specs)[0]]
            
            user_input = message

              # Instantiate LangChainAgent and call run_agent_and_llms
            agent = LangChainAgent()
            results, formatted_label_str = asyncio.run(agent.run_agent_and_llms(
                model_names,
                self.model_specs,
                temp_slider,
                "System: " + self.sys_msg_text,
                [new_chat_history] + new_model_chatbots,
                user_input
            ))
            first_model = model_names[0]
            chat_history[-1] = (chat_history[-1][0], self.model_specs[first_model]['friendly_name'] + ": " + results[first_model])

            for i, model_spec_key in enumerate(list(self.model_specs.keys())[0:], start=0):
                if self.model_specs[model_spec_key]['is_used']:
                    friendly_name = self.model_specs[model_spec_key]['friendly_name']
                    if model_spec_key in results:
                        model_chatbots[i][-1] = (model_chatbots[i][-1][0], friendly_name + ": " + results[model_spec_key])
                    else:
                        model_chatbots[i][-1] = (model_chatbots[i][-1][0], self.model_specs[first_model]['friendly_name'] + ": " + results[first_model])

            return "", chat_history, message_selected, formatted_label_str, *model_chatbots
        else:
            for i, (msg_index, msg_text) in enumerate(chat_history):
                iDelta = len(chat_history) - len(model_chatbots[0])
                if msg_index == message_selected or msg_text == message_selected:
                    if message:
                        chat_history[i] = (msg_index, message)
                        for model_chatbot in model_chatbots:
                            model_chatbot[i-iDelta] = (msg_index, message)
                    else:
                        chat_history.pop(i-iDelta)
                        for model_chatbot in model_chatbots:
                            model_chatbot.pop(i-iDelta)
                    break
            message_selected = None
            return "", chat_history, message_selected, perf_label, *model_chatbots