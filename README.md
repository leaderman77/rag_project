# SmartRetrieval RAG Project

This project implements a RAG-powered Question Answering application, focusing on optimizing retrieval techniques to enhance response accuracy and contextual relevance. It employs the LlamaIndex framework with its in-built vector store and Neo4j for knowledge management to assess various advanced retrieval methods.

## Features

- Implements multiple retrieval methods:
  - Standard base retrieval
  - Sentence window retrieval
  - Auto-merging retrieval
  - Knowledge graph-based retrieval

- Uses a dataset of 10 research papers focused on AI and LLMs
- Includes a Gradio web application for model testing
- Provides a FastAPI web service for model integration
- Docker support for easy deployment and scaling
- Performance evaluation using the Tonic Validate framework

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API Key
- (Optional) Tonic Validate API Key, Product Key, and Benchmark Key for evaluation

## Installation and Usage

### Option 2: Using Docker (Recommended)
1. Clone the repository:
git clone https://github.com/your-username/smartretrieval-rag.git
cd smartretrieval-rag

2. Create a .env file in the root directory and add your OpenAI API Key:
OPENAI_API_KEY=your_openai_api_key_here

3. Build and run the Docker container:

docker-compose up --build

### Option 2: Using Conda Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/smartretrieval-rag.git
   cd smartretrieval-rag
2. Create and activate the Conda environment:
   conda env create -f environment.yml
   conda activate rag_project_env
3. Create a .env file in the root directory and add your OpenAI API Key:
   OPENAI_API_KEY=your_openai_api_key_here
4. Run the application:
   python main.py


Access the Gradio interface at [http://localhost:8000](http://localhost:8000/) in your web browser.
To use the FastAPI service, send a POST request to http://localhost:8000/ask as described in Option 1.

### Evaluation
To evaluate the performance of RAG retrievals using the Tonic Validate framework:

1. Add the following keys to your .env file:
  TONIC_VALIDATE_API_KEY=your_tonic_validate_api_key
  TONIC_VALIDATE_PRODUCT_KEY=your_tonic_validate_product_key
  TONIC_VALIDATE_BENCHMARK_KEY=your_tonic_validate_benchmark_key  # Optional
2. Run the evaluation script (implementation details to be provided).

### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
