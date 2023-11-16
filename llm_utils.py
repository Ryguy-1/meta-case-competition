from langchain.llms.ollama import Ollama
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from typing import List

# abstract class for all llm models
from abc import ABC, abstractmethod


class OllamaModel(ABC):
    """Abstract class for Ollama Models"""

    def __init__(self, ollama_model_name: str) -> None:
        """
        Initialize the Ollama Model.

        Params:
            ollama_model_name (str): The Ollama model name to use.
        """
        self.ollama_model_name = ollama_model_name
        self.llm = Ollama(
            model=ollama_model_name, temperature=0.5
        )  # cold hearted, but still slight bit of randomness

    @abstractmethod
    def run(self, instructions: str) -> str:
        """
        Write Section According to Instructions.

        Params:
            instructions (str): Instructions for rewriting.

        Returns:
            str: Rewritten section.
        """
        pass
