from typing import Any, List, Mapping, Optional
from pydantic import Field
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from dmlphi3.phi3dml import generate_response as dmlphi3response
try:
    from dmltorch.dmltorch import LLM_Model
except ImportError:
    # Handle the import error or pass if you want to ignore it
    print("Warning: LLM_Model could not be imported. This code path will be skipped.")

import sys
import os
# Ensure subfolder is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dmltorch'))

class LocalModelInterface(LLM):
    model_type: str = Field(default="llmt1")
    max_output_tokens: int = Field(default=100)
    temperature: float = Field(default=0.7)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if self.model_type == "OnnxDML: phi3":
            response = self._local_OnnxDMLPhi3(prompt)
        elif self.model_type == "TorchDML: phi3":
            response = self._local_TorchDMLPhi3(prompt)
        elif self.model_type == "local-model2":
            response = self._local_model2(prompt)
        else:
            raise ValueError(f"Unsupported local model: {self.model_type}")
        return response

    def _local_OnnxDMLPhi3(self, prompt: str) -> str:
        # Implementation of local model 1
        # Use self.max_output_tokens and self.temperature as needed
        # ...
        response = dmlphi3response(prompt, self.max_output_tokens, self.temperature)
        return response

    def _local_TorchDMLPhi3(self, prompt: str) -> str:
        # Implementation of local model 1
        # Use self.max_output_tokens and self.temperature as needed
        llm_model = LLM_Model(
            prompt="Hello",
            interactive=False,
            num_samples=1,
            max_new_tokens=self.max_output_tokens,
            top_k=200,
            temperature=self.temperature,
        )
        llm_model.load_model()
        
        # Call the chat method and accumulate the full response
        response = llm_model.chat(prompt, "")
        total_msg = ""
        for msg in response:
            total_msg += msg
            
        return total_msg

    def _local_model2(self, prompt: str) -> str:
        # Implementation of local model 2
        # Use self.max_output_tokens and self.temperature as needed
        # ...
        response = "This is a sample response"
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_type": self.model_type}

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _generate(
        self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> LLMResult:
        if len(prompts) > 1:
            raise ValueError("LocalModelInterface only supports single prompts.")
        
        # Extract the input string from the prompts list
        if isinstance(prompts[0], dict):
            input_str = prompts[0]["input"]
        else:
            input_str = prompts[0]
        
        # Generate the response using the _call method
        response = self._call(input_str, stop=stop)
        
        # Create an LLMResult object with the generated response
        generation = Generation(text=response)
        generations = [[generation]]  # Wrap the Generation object inside a list of lists
        llm_output = {"model_type": self.model_type}
        return LLMResult(generations=generations, llm_output=llm_output)