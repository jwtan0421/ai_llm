from langchain_core.runnables.utils import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import QianfanChatEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from prompt_base import prompt_template
import os

class LLMUtils:
    """
    A utility class for handling multiple language models and invoking them based on user input.
    """

    def __init__(self, model_name="gpt"):
        """
        Initialize the LLMUtils class with model configurations.

        Args:
            model_name (str): The default model to use. Defaults to "gpt".
        """
        self.model_name = model_name        

        # Initialize QianfanChatEndpoint with credentials from environment variables
        self.ernie_model = QianfanChatEndpoint(
            qianfan_ak=os.getenv('ERNIE_CLIENT_ID'),
            qianfan_sk=os.getenv('ERNIE_CLIENT_SECRET')
        )

        # Initialize GPT model
        self.gpt_model = ChatOpenAI()

        # Select model based on configurable alternatives
        self.model = self.gpt_model.configurable_alternatives(
            ConfigurableField(id="llm"), 
            default_key="gpt", 
            ernie=self.ernie_model,
            # Additional models can be added here
        )

        # Load the prompt template
        self.prompt = ChatPromptTemplate.from_template(prompt_template)

    def invoke(self, question, context_retriever):
        """
        Invoke the language model to get an answer to the given question using the specified context retriever.

        Args:
            question (str): The question to be answered.
            context_retriever (object): The retriever object to get relevant context documents.

        Returns:
            tuple: A tuple containing the model response and the relevant texts.
        """
        self.context_retriever = context_retriever
        
        # Retrieve relevant documents based on the question
        ref_docs = self.context_retriever.invoke(question)     
        relevant_texts = [doc.page_content for doc in ref_docs]
        relevant_texts = "\n\n".join(relevant_texts)

        # Define the processing chain
        self.chain = (
            {"question": RunnablePassthrough(), "context": self.context_retriever}
            | self.prompt
            | self.model 
            | StrOutputParser()
        )

        # Invoke the chain with the specified model configuration
        response = self.chain.with_config(configurable={"llm": self.model_name}).invoke(question)
        
        return response, relevant_texts
    