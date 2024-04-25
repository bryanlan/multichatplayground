model_specs = {
    "claude-3-opus-20240229": {
        "is_local":False,
        "maximum_context_length_tokens": 200000,
        "maximum_output_tokens": 4096,
        "friendly_name":"Claude",
        "cost_per_token": {
            "input": 0.015,  # $0.015 per 1k tokens
            "output": 0.075  # $0.075 per 1k tokens
        }
    },
    "gpt-4-turbo-preview": {
        "is_local":False,
        "maximum_context_length_tokens": 128000,
        "maximum_output_tokens": 4096,
        "friendly_name":"GPT4",
        "cost_per_token": {
            "input": 0.01,  # $0.01 per 1k tokens
            "output": 0.03  # $0.03 per 1k tokens
        }
    },
    "gemini-pro": {
        "is_local":False,
        "maximum_context_length_tokens": 128000,
        "maximum_output_tokens": 2048,
        "friendly_name":"Gemini",
        "cost_per_token": {
            "input": 0.001,  # $0.001 per 1k tokens
            "output": 0.002  # $0.002 per 1k tokens
        }
    },
    "Mixtral-8x7B-Instruct-v0.1": {
        "is_local":False,
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name":"Mixtral",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    },
    "Ollama: llama3": {
        "is_local":True,
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name":"Ollama: llama3",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    },
    "Ollama: phi": {
        "is_local":True,
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name":"Ollama: phi",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    },
     "Ollama: phi3": {
        "is_local":True,
        "maximum_context_length_tokens": 2048,
        "maximum_output_tokens": 2048,
        "friendly_name":"Ollama: phi3",
        "cost_per_token": {
            "input": 0.000,  # $0.001 per 1k tokens
            "output": 0.000  # $0.002 per 1k tokens
        }
    }
}
