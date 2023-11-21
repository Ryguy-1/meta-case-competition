from langchain.llms.ollama import Ollama
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from typing import List
import pandas as pd


def main() -> None:
    """Main function for running the Ollama model."""
    ollama = OllamaKeywordFinder("mistral-openorca")
    df = pd.read_csv("docs_per_topic.csv")
    # get overview column where cluster_labels column is 57
    overview = df[df["cluster_labels"] == 57]["overview"].tolist()[0]
    overview = overview.split(" ")
    print(overview)
    category = ollama.run(overview)
    print(category)


class OllamaKeywordFinder(object):
    """Ollama Model for Finding Keywords"""

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

    def run(self, keyword_list: List[str]) -> str:
        """
        Write Category from Keywords.

        Params:
            keyword_list (List[str]): The list of keywords to write a category from.

        Returns:
            str: The category written from the keywords.
        """
        write_template = PromptTemplate.from_template(
            template="""
                You are an AI tasked with programmatically deciding a category to put a list of keywords into.
                You are in a code pipeline, and you are given just the keywords to work with.
                Any text you output will be taken as the category for the keywords exactly and used later in the pipeline.
                You will be a reliable and trusted part of the pipeline, only outputting as told to do so.
                Stick as closely to the keywords as possible when you write the category.
                The category should just be a short phrase, similar to a genre.

                The keyword list is: "{inst}"
                Your final genre choice: """,
        )
        chain = write_template | self.llm | StrOutputParser()
        output = chain.invoke({"inst": str(keyword_list)})
        output = output.strip()
        output = output.lower()
        output = output[0].upper() + output[1:]  # capitalize
        return output


if __name__ == "__main__":
    main()
