model_specs = {
      "gpt-4o": {
        "is_local": False,
        "supports_agents":True,
        "is_used": True,  # Add this line and set to True or False as needed
        "maximum_context_length_tokens": 128000,
        "maximum_output_tokens": 4096,
        "friendly_name": "GPT4o",
        "cost_per_token": {
            "input": 0.01,  # $0.01 per 1k tokens
            "output": 0.03  # $0.03 per 1k tokens
        }
    },
    "claude-3-opus-20240229": {
        "is_local": False,
        "supports_agents":True,
        "is_used": True,  # Add this line and set to True or False as needed
        "maximum_context_length_tokens": 200000,
        "maximum_output_tokens": 4096,
        "friendly_name": "Claude",
        "cost_per_token": {
            "input": 0.015,  # $0.015 per 1k tokens
            "output": 0.075  # $0.075 per 1k tokens
        }
    },
  
    "gemini-pro": {
        "is_local": False,
        "supports_agents":False,
        "is_used": True,  # Add this line and set to True or False as needed
        "maximum_context_length_tokens": 128000,
        "maximum_output_tokens": 2048,
        "friendly_name": "Gemini",
        "cost_per_token": {
            "input": 0.001,  # $0.001 per 1k tokens
            "output": 0.002  # $0.002 per 1k tokens
        }
    },
    "Mixtral-8x7B-Instruct-v0.1": {
        "is_local": False,
        "supports_agents":False,
        "is_used": True,  # Add this line and set to True or False as needed
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name": "Mixtral",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    },
    "Ollama: llama3": {
        "is_local": True,
        "supports_agents":False,
        "is_used": True,  # Add this line and set to True or False as needed
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name": "Ollama: llama3",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    },
    "Ollama: phi": {
        "is_local": True,
        "supports_agents":False,
        "is_used": True,  # Add this line and set to True or False as needed
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name": "Ollama: phi",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    },
    "Ollama: phi3": {
        "is_local": True,
        "supports_agents":False,
        "is_used": True,  # Add this line and set to True or False as needed
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name": "Ollama: phi3",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    },
    # "OnnxDML: phi3": {
    #     "is_local":True,
    #     "supports_agents":False,
    #     "is_used": False,  # Disabled due to missing onnxruntime_genai
    #     "maximum_context_length_tokens": 2048,
    #     "maximum_output_tokens": 2048,
    #     "friendly_name":"OnnxDML: phi3",
    #     "cost_per_token": {
    #         "input": 0.000,  # $0.001 per 1k tokens
    #         "output": 0.000  # $0.002 per 1k tokens
    #     }
    # },
    #  "TorchDML: phi3": {
    #     "is_local":True,
    #     "supports_agents":False,
    #     "is_used": False,  # Disabled due to missing onnxruntime_genai
    #     "maximum_context_length_tokens": 2048,
    #     "maximum_output_tokens": 2048,
    #     "friendly_name":"TorchDML: phi3",
    #     "cost_per_token": {
    #         "input": 0.000,  # $0.001 per 1k tokens
    #         "output": 0.000  # $0.002 per 1k tokens
    #     }
    # },
    "gpt-image-1": {
        "is_local": False,
        "supports_agents": False,
        "supports_images": True,
        "is_used": True,
        "maximum_context_length_tokens": 0,  # Not applicable for image generation
        "maximum_output_tokens": 0,  # Not applicable for image generation
        "friendly_name": "GPT Image Generator",
        "cost_per_token": {
            "input": 0.005,  # $5 per 1M tokens
            "output": 0.040  # $40 per 1M image tokens (~$0.19 per image)
        }
    },
    "imagen-3.0-generate-002": {
        "is_local": False,
        "supports_agents": False,
        "supports_images": True,
        "is_used": True,
        "maximum_context_length_tokens": 0,  # Not applicable for image generation
        "maximum_output_tokens": 0,  # Not applicable for image generation
        "friendly_name": "Gemini Image Generator",
        "cost_per_token": {
            "input": 0.000,  # Free tier or minimal cost
            "output": 0.000  # Free tier or minimal cost
        }
    },
}
