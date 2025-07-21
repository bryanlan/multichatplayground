Multichatplayground is a cool wrapper for Gradio that makes it super easy to build chatbot demos that use multiple local and remote models concurrently, now with **dual image generation support** from OpenAI and Google Gemini!

## ✨ New Features
- **🎨 Image Generation**: Generate images using OpenAI's `gpt-image-1` and Google's `imagen-3.0-generate-002` models
- **🖼️ Individual Model Tabs**: View generated images in dedicated tabs for each provider
- **🎯 Smart Model Filtering**: "Image Generation Models Only" checkbox to focus on image-capable models
- **⚡ Concurrent Generation**: Generate images from multiple providers simultaneously

<img width="791" alt="image" src="https://github.com/bryanlan/multichatplayground/assets/1688838/62d21aee-59c2-469d-bda2-2bca79d30c0a">
<img width="770" alt="image" src="https://github.com/bryanlan/multichatplayground/assets/1688838/04c31c58-6750-48fb-a44e-7e8936ed2107">
<img width="779" alt="image" src="https://github.com/bryanlan/multichatplayground/assets/1688838/51b6958f-c5cb-4c66-a626-25c19aa65758">

## Features

### 💬 Multi-Model Chat
The user can enter in a prompt once and it will concurrently execute on all models, using langchain to maintain appropriate conversational content. For non-local models, API keys will be required, and local models use ollama to execute. 

### 🎨 Image Generation
- **OpenAI Integration**: Uses `gpt-image-1` model for high-quality image generation
- **Google Gemini Integration**: Uses `imagen-3.0-generate-002` model via predict API
- **Dual Provider Support**: Generate images from both providers simultaneously
- **Individual Display**: Each generated image appears in its respective model tab

More models can easily be added by adding to the modelindex.py function. In addition, conversations can be saved, loaded and removed by changing the name of the active chat. 
<img width="808" alt="image" src="https://github.com/bryanlan/multichatplayground/assets/1688838/72e13357-8194-445d-ae87-7cce21a2055f">


Further, this building tool has playground like functionality where the user can go back and edit a previous response of the agent and have that updated context be used for the next prompt. For example:

<img width="757" alt="image" src="https://github.com/bryanlan/multichatplayground/assets/1688838/0551ef13-fb49-4d22-ad20-44fcc680c264">


## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/bryanlan/multichatplayground.git
cd multichatplayground

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements_clean.txt
```

### 2. Setup API Keys
```bash
# Copy the template and add your API keys
cp keys_dummy.py keys.py

# Edit keys.py with your actual API keys:
# - OPENAI_API_KEY: For GPT models and image generation
# - GOOGLE_API_KEY: For Gemini models and image generation
# - ANTHROPIC_API_KEY: For Claude models (optional)
# - HUGGINGFACE_API_KEY: For Mixtral models (optional)
```

### 3. Launch the Application
```bash
python gradiosample.py
```
The app will be available at `http://localhost:7865`

## 🎨 Using Image Generation

1. **Enable Image Mode**: Check "Image Generation Models Only" to filter models
2. **Select Models**: Choose "GPT Image Generator" and/or "Gemini Image Generator"  
3. **Enter Prompt**: Describe the image you want (e.g., "A sunset over mountains with vibrant colors")
4. **View Results**: Generated images appear in their respective model tabs

## 💻 Code Integration

The sample code can be found at gradiosample.py.

After setting up the UI code according to preference, the only additional code required for this multi-chatbox functionality is:
```python
  ThreadText = ChatBot(full_path = "savedChats.json", json_object = {},full_json_object = {}, name_text_box=activeThread,
                         dropdown = thread_choice, clear_button= clear_chat, save_button = save_button, remove_button = remove_button,
                         primary_chatbot=chatbot,secondary_chatbots=model_chatbots, model_specs = model_specs, model_choice=model_choice, 
                         temp_slider=temp_slider, msg = msg,sys_msg_text= "System: You are a helpful assistant")
  ThreadText.bind_event_handlers()
  ThreadText.initialize()
```
