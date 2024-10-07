import os
import sys
import logging
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    get_leaf_nodes,
    get_root_nodes
)
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    KnowledgeGraphIndex
)
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def build_base_index(documents, save_dir_base_index="base_index"):
    """
    Processes documents by splitting them into sentences, indexing them, and either saving the index
    to a directory or loading it if it already exists.

    Parameters:
    documents (list): List of document strings to process.
    save_dir_base_index (str): Directory where the index is saved or to be saved.

    Returns:
    VectorStoreIndex: The base index created from the documents or loaded from the storage.
    """
    # Splitting the documents into base nodes (sentences)
    base_nodes = SentenceSplitter().get_nodes_from_documents(documents)

    # Save the base index from the specified directory
    base_index = VectorStoreIndex(base_nodes, show_progress=True)
    base_index.storage_context.persist(persist_dir=save_dir_base_index)

    print("BASE INDEX SAVED!!!")
    return base_index


def process_base_index(documents, save_dir_base_index):
    # Save or load the base index from the specified directory
    if not os.path.exists(save_dir_base_index):
        base_index = build_base_index(documents, save_dir_base_index)
    else:
        # load base index from db
        base_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir_base_index), show_progress=True)
        print("BASE INDEX LOADED SUCCESSFULLY!!!")
    return base_index


def build_sentence_window_index(
        documents,
        sentence_window_size=6,
        save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    sentence_nodes = node_parser.get_nodes_from_documents(documents)

    sentence_index = VectorStoreIndex(sentence_nodes, show_progress=True)
    sentence_index.storage_context.persist(persist_dir=save_dir)
    print("SENTENCE INDEX SAVED!!!")
    return sentence_index


def process_sentence_window_index(
        documents,
        sentence_window_size=6,
        save_dir="sentence_index",
):
    # Save or load the sentence window index from the specified directory
    if not os.path.exists(save_dir):
        sentence_index = build_sentence_window_index(documents, sentence_window_size, save_dir)
    else:
        # load sentence index from db
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir), show_progress=True)
        print("SENTENCE INDEX LOADED SUCCESSFULLY!!!")
    return sentence_index


def get_sentence_window_query_engine(
        sentence_index,
        llm,
        embed_model,
        prompt_template,
        similarity_top_k=6,
        rerank_top_n=2
):
    # define postprocessors
    post_proc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    print("SENTENCE RERANK LOADED!!!")
    sentence_window_engine = sentence_index.as_query_engine(
        text_qa_template=prompt_template, similarity_top_k=similarity_top_k, embed_model=embed_model,
        llm=llm, node_postprocessors=[post_proc, rerank]
    )
    return sentence_window_engine


def build_auto_merging_retriever(documents, save_dir="auto_merge_index"):
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 512, 256])
    nodes = node_parser.get_nodes_from_documents(documents)
    print("Nodes:", len(nodes))

    leaf_nodes = get_leaf_nodes(nodes)
    print("Leaf Nodes: ", len(leaf_nodes))

    root_nodes = get_root_nodes(nodes)
    print("Root nodes: ", len(root_nodes))

    docstore = SimpleDocumentStore()
    # insert nodes into docstore
    docstore.add_documents(nodes)

    # define storage context (will include vector store by default too)
    storage_context = StorageContext.from_defaults(docstore=docstore)

    # save index into db
    auto_merging_index = VectorStoreIndex(
        leaf_nodes, storage_context=storage_context, show_progress=True
    )
    auto_merging_index.storage_context.persist(persist_dir=save_dir)
    print("AUTO-MERGE INDEX SAVED!!!")
    return auto_merging_index


def process_auto_merging_index(documents, save_dir):
    # Save or load the auto-merging index from the specified directory
    if not os.path.exists(save_dir):
        auto_merging_index = build_auto_merging_retriever(documents, save_dir)
    else:
        # load sentence index from db
        auto_merging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir), show_progress=True)
        print("AUTO-MERGE INDEX LOADED SUCCESSFULLY!!!")
    return auto_merging_index


def build_knowledge_graph(documents, save_dir="kg_index"):
    # Neo4j Graph Store Setup
    graph_store = Neo4jGraphStore(username="neo4j",
                                  password="Ss123456$",
                                  url="bolt://localhost:7687",  # "bolt://7.tcp.eu.ngrok.io:18000",
                                  database="neo4j")

    storage_context = StorageContext.from_defaults(graph_store=graph_store)

    kg_index = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=10,
        include_embeddings=True,
        show_progress=True
    )
    # save and load
    kg_index.storage_context.persist(persist_dir=save_dir)
    print("KNOWLEDGE GRAPH INDEX SAVED!!!")
    return kg_index


def process_knowledge_graph_index(documents, save_dir):
    # Save or load the auto-merging index from the specified directory
    if not os.path.exists(save_dir):
        kg_index = build_knowledge_graph(documents, save_dir)
    else:
        # load sentence index from db
        kg_index = load_index_from_storage(StorageContext.from_defaults(
            persist_dir="kg_index"), show_progress=True)
        print("KNOWLEDGE GRAPH INDEX LOADED SUCCESSFULLY!!!")
    return kg_index
