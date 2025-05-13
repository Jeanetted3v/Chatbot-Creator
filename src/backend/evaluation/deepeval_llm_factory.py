"""To run:
python -m src.backend.evaluation.deepeval_llm_factory
"""

from typing import Optional, Dict, Any, Union
from deepeval.models.base_model import DeepEvalBaseLLM

# Import necessary models
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from src.backend.utils.settings import SETTINGS


class DeepEvalLLMFactory:
    """Factory class to create and manage different LLM models for DeepEval."""
    
    @staticmethod
    def create_llm(provider: str, **kwargs) -> DeepEvalBaseLLM:
        """Create and return an LLM model wrapped for DeepEval.
        
        Args:
            provider (str): The LLM provider to use ('azure', 'groq', etc.)
            **kwargs: Provider-specific parameters
            
        Returns:
            DeepEvalBaseLLM: A wrapped LLM model ready for use with DeepEval
            
        Raises:
            ValueError: If an unsupported provider is specified
        """
        if provider.lower() == "azure":
            return AzureOpenAIWrapper(**kwargs)
        if provider.lower() == "groq":
            return GroqWrapper(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class AzureOpenAIWrapper(DeepEvalBaseLLM):
    """Wrapper for Azure OpenAI models compatible with DeepEval."""
    
    def __init__(
        self,
        azure_deployment: str,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2023-05-15",
        **kwargs
    ):
        self.azure_deployment = azure_deployment
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = api_key
        self.azure_api_version = api_version
        self.additional_kwargs = kwargs
        self._model = None
    
    def load_model(self):
        if self._model is None:
            self._model = AzureChatOpenAI(
                azure_deployment=self.azure_deployment,
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                **self.additional_kwargs
            )
        return self._model
    
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
    
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content
    
    def get_model_name(self):
        return f"Azure OpenAI ({self.azure_deployment})"


class GroqWrapper(DeepEvalBaseLLM):
    """Wrapper for Groq models compatible with DeepEval."""
    
    def __init__(
        self,
        model_name: str = "llama3-70b-8192",
        groq_api_key: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name
        self.groq_api_key = groq_api_key
        self.additional_kwargs = kwargs
        self._model = None
    
    def load_model(self):
        if self._model is None:
            self._model = ChatGroq(
                model_name=self.model_name,
                groq_api_key=self.groq_api_key,
                **self.additional_kwargs
            )
        return self._model
    
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
    
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content
    
    def get_model_name(self):
        return f"Groq ({self.model_name})"


# Test usages
if __name__ == "__main__":
    # Azure OpenAI example
    azure_openai = DeepEvalLLMFactory.create_llm(
        provider="azure",
        azure_deployment="gpt-4o",
        azure_endpoint=SETTINGS.AZURE_ENDPOINT,
        api_key=SETTINGS.AZURE_API_KEY,
        api_version=SETTINGS.AZURE_API_VERSION,
    )
    
    # Groq example
    groq_llm = DeepEvalLLMFactory.create_llm(
        provider="groq",
        model_name="llama3-70b-8192",
        api_key=SETTINGS.GROQ_API_KEY
    )
    
    # Generate responses
    print("Azure OpenAI response:", azure_openai.generate("Write me a joke"))
    print("Groq response:", groq_llm.generate("Write me a joke"))