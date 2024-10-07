import os
import openai
from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from process_retriever_index import get_sentence_window_query_engine


class RAGEngine:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Set the OpenAI API key for authentication.
        openai.api_key = os.getenv("OPENAI_API_KEY")

        if openai.api_key is None:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        # Set up LLM and embedding model
        self.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Load indexes
        self.base_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="storage/base_index"))
        self.sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="storage/sentence_index"))
        self.auto_merging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="storage/auto_index"))
        self.knowledge_graph_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="storage/kg_index"))

        # Load prompt template
        with open("resources/text_qa_template.txt", 'r', encoding='utf-8') as file:
            self.prompt_template = PromptTemplate(file.read())

    def get_query_engine(self, retriever_type):
        if retriever_type == 'base':
            return self.base_index.as_query_engine()
        elif retriever_type == 'sentence_window':
            return get_sentence_window_query_engine(
                self.sentence_index,
                self.llm,
                self.embed_model,
                self.prompt_template
            )
        elif retriever_type == 'auto_merging':
            auto_base_retriever = self.auto_merging_index.as_retriever(similarity_top_k=6)
            return RetrieverQueryEngine.from_args(
                AutoMergingRetriever(auto_base_retriever, self.auto_merging_index.storage_context, verbose=True)
            )
        elif retriever_type == 'knowledge_graph':
            return self.knowledge_graph_index.as_query_engine(
                include_text=True,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=5
            )
        else:
            raise ValueError("Invalid retriever type")

    async def ask_question(self, question: str, retriever_type: str) -> str:
        query_engine = self.get_query_engine(retriever_type)
        response = query_engine.query(question)
        return str(response)
