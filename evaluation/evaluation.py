from llama_index.core import Settings
import os
import nest_asyncio
import warnings
import logging
import pandas as pd
from llama_index.core import StorageContext
from utils import run_experiment, load_config
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from tonic_validate import ValidateScorer, ValidateApi, Benchmark
from llama_index.core.retrievers import AutoMergingRetriever
from tonic_validate.metrics import (
    RetrievalPrecisionMetric,
    AnswerSimilarityMetric,
    AnswerConsistencyMetric,
    # AugmentationPrecisionMetric,
    # AugmentationAccuracyMetric,
    LatencyMetric,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from process_retriever_index import get_sentence_window_query_engine

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    nest_asyncio.apply()
    warnings.filterwarnings('ignore')


def initialize_environment():
    load_config()
    os.environ["TONIC_VALIDATE_API_KEY"] = os.getenv("TONIC_VALIDATE_API_KEY")
    os.environ["TONIC_VALIDATE_PROJECT_KEY"] = os.getenv("TONIC_VALIDATE_PROJECT_KEY")
    os.environ["TONIC_VALIDATE_BENCHMARK_KEY"] = os.getenv("TOMIC_VALIDATE_BENCHMARK_KEY")
    api_keys = {
        'openai': os.getenv("OPENAI_API_KEY"),
        'tonic_validate': os.getenv("TONIC_VALIDATE_API_KEY"),
        'project_key': os.getenv("TONIC_VALIDATE_PROJECT_KEY"),
        'benchmark_key': os.getenv("TONIC_VALIDATE_BENCHMARK_KEY")
    }
    validate_api = ValidateApi(api_keys['tonic_validate'])
    print(validate_api)
    print(api_keys)
    return validate_api, api_keys


def setup_llm_and_embeddings():
    llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def setup_validate_scorer(validate_api, tonic_validate_benchmark_key):
    benchmark = validate_api.get_benchmark(tonic_validate_benchmark_key)
    return ValidateScorer(
        metrics=[RetrievalPrecisionMetric(),
                 AnswerSimilarityMetric(),
                 AnswerConsistencyMetric(),
                 # AugmentationPrecisionMetric(),
                 # AugmentationAccuracyMetric(),
                 LatencyMetric()],
        # gpt-3.5-turbo gemini-1.5-flash gpt-4o-mini
        model_evaluator="gpt-4o"), benchmark


def load_query_engines(
        base_index, sentence_index, auto_merging_index,
        knowledge_graph_index, llm, embed_model, prompt_template):
    """
        Initializes and loads various query engines using different configurations.
        This function demonstrates handling for a base index, a sentence window retrieval,
        and several composite engines with transformations and rerankers.

        Parameters:
        llm (OpenAI): The language model object.
        embed_model (OpenAIEmbedding): The embedding model object.
        prompt_template (PromptTemplate): The prompt template for query engines.

        Returns:
        dict: A dictionary containing initialized query engines.
        """
    try:
        engines = {}

        # naive RAG
        naive_rag_engine = base_index.as_query_engine(
            llm=llm, text_qa_template=prompt_template, similarity_top_k=3, embed_model=embed_model
        )
        engines['Naive RAG'] = naive_rag_engine

        # sentence window
        sentence_window_engine = sentence_index.as_query_engine(
            text_qa_template=prompt_template, similarity_top_k=3, embed_model=embed_model, llm=llm
        )
        engines['Sentence window retrieval'] = sentence_window_engine

        # Sentence window retrieval + Sentence Transformer rerank
        engines['Sentence window retrieval + Sentence rerank'] = get_sentence_window_query_engine(
            sentence_index, llm, embed_model, prompt_template, similarity_top_k=6, rerank_top_n=2
        )

        auto_base_retriever = auto_merging_index.as_retriever(similarity_top_k=6)
        engines['Auto-merging retrieval'] = RetrieverQueryEngine.from_args(
            AutoMergingRetriever(auto_base_retriever,
                                 StorageContext.from_defaults(persist_dir="auto_index"),
                                 verbose=True))

        engines['Knowledge graph based retrieval'] = knowledge_graph_index.as_query_engine(
            text_qa_template=prompt_template, include_text=True, response_mode="tree_summarize",
            embedding_mode="hybrid", similarity_top_k=5)

        return engines
    except Exception as e:
        logging.error(f"Error initializing query engines: {e}")
        raise


def load_benchmark():
    import json
    # Load questions and answers from benchmark.json
    with open('../eval_questions/benchmark.json', 'r') as file:
        benchmark_data = json.load(file)
        questions = benchmark_data['questions']
        ground_truths = benchmark_data['ground_truths']

    return questions, ground_truths


def run_experiments(experiments, scorer, benchmark, validate_api):
    results_df = pd.DataFrame(columns=['Run', 'Experiment', 'OverallScores'])
    for name, engine in experiments.items():
        try:
            results = run_experiment(name, engine, scorer,
                                     benchmark, validate_api,
                                     "benchmark_id", upload_results=True, runs=3)
            results_df = pd.concat([results_df, results], ignore_index=True)
        except Exception as e:
            logging.error(f"Error running experiment {name}: {e}")
    return results_df


def run_evaluation(
        base_index, sentence_index, auto_merging_index,
        knowledge_graph_index, llm, embed_model, prompt_template
):
    setup_logging()
    validate_api, api_keys = initialize_environment()
    scorer, benchmark = setup_validate_scorer(validate_api, api_keys['benchmark_key'])
    experiments = load_query_engines(
        base_index, sentence_index, auto_merging_index, knowledge_graph_index,
        llm, embed_model, prompt_template
    )
    questions, ground_truths = load_benchmark()
    # Create the Benchmark instance
    benchmark = Benchmark(questions=questions[:10], answers=ground_truths[:10])
    results_df = run_experiments(experiments, scorer, benchmark, validate_api)

    return results_df