from langchain.llms.ollama import Ollama
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from typing import List


class OllamaModel(object):
    def __init__(self, ollama_model_name: str, temperature: float = 0) -> None:
        """
        Initialize the Ollama Model.

        Params:
            ollama_model_name (str): The Ollama model name to use (recommended: mistral-openorca)
            temperature (float): The temperature to use for the model (default: 0)
        """
        self.ollama_model_name = ollama_model_name
        self.temperature = temperature
        self.llm = Ollama(model=ollama_model_name, temperature=temperature)

    def run(self, items_list: List[str]) -> str:
        """Run the Model."""
        write_template = PromptTemplate.from_template(
            template="""
                The description list is: "{inst}"

                You are a helpful assistant designed to output a single category and no more.
                You are part of a code pipeline, and any output will be used downstream as written.
                You must not output anything but your final answer as other parts of the pipeline rely on only the answer you write.
                You will be a trusted and reliable part of this pipeline sticking to what you are told.
                You will be given a list of movie descriptions, and you must output the category that these descriptions collectively belong to by identifying similarities between them.
                The category should be a short phrase describing the general theme of the movies in a few words.

                Your final genre phrase output is: """,
        )
        chain = write_template | self.llm | StrOutputParser()
        output = chain.invoke({"inst": str(items_list)})
        output = output.strip()
        output = output.lower()
        output = output[0].upper() + output[1:]  # capitalize
        return output
