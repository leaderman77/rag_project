# SmartRetrieval RAG Project

This project implements a RAG-powered Question Answering application, focusing on optimizing retrieval techniques to enhance response accuracy and contextual relevance. It employs the LlamaIndex framework with its in-built vector store and Neo4j for knowledge management to assess various advanced retrieval methods.

### Table of Contents
- Features
- Prerequisites
- Installation and Usage
- Option 1: Using Docker (Recommended)
- Option 2: Using Conda Environment
- Project Structure
- Evaluation
- Troubleshooting
- Contributing
- License
  
## Features

- Implements multiple retrieval methods:
  - Standard base retrieval
  - Sentence window retrieval
  - Auto-merging retrieval
  - Knowledge graph-based retrieval

- **Rich Dataset**: Utilizes 10 research papers focused on AI and LLMs
- **User-Friendly Interface**: Gradio web application for easy model testing
- **API Integration**: FastAPI web service for seamless model integration
- **Scalability**: Docker support for effortless deployment and scaling
- **Performance Metrics**: Evaluation using the Tonic Validate framework

## Prerequisites

- Python 3.12+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API Key
- (Optional) Tonic Validate API Key, Product Key, and Benchmark Key for evaluation

## Installation and Usage

### Option 1: Using Docker (Recommended)
1. Clone the repository:
   git clone https://github.com/your-username/smartretrieval-rag.git
   cd smartretrieval-rag

2. Provide your OpenAI API Key in .env file located in the root directory:
   OPENAI_API_KEY=your_openai_api_key_here

3. Build and run the Docker container:
   docker-compose up --build

### Option 2: Using Conda Environment

1. Clone the repository:
   git clone https://github.com/your-username/smartretrieval-rag.git
   cd smartretrieval-rag
2. Create and activate the Conda environment:
   conda env create -f environment.yml
   conda activate rag_project_env
3. Create a .env file in the root directory and add your OpenAI API Key:
   OPENAI_API_KEY=your_openai_api_key_here
4. Run the application:
   python main.py

### Access to Gradio and FastAPI
Access the Gradio interface at [http://localhost:8000](http://localhost:8000/) in your web browser.
To use the FastAPI service, send a POST request to http://localhost:8000/ask as described in Option 1.

## Project Structure
smartretrieval-rag/
│
├── main.py                 # Entry point for the application
├── app/
│   ├── api.py              # FastAPI routes and endpoints
│   ├── gradio_interface.py # Gradio web interface
│   └── rag/
│       ├── indexing.py     # Vector and knowledge graph indexing
│       └── retrieval.py    # Retrieval methods implementation
├── data/
│   └── papers/             # AI and LLM research papers
├── storage/                # Indexed data storage
├── resources/              # Additional resources
├── tests/                  # Unit and integration tests
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── environment.yml         # Conda environment specification
└── requirements.txt        # Python dependencies

### Evaluation
To evaluate the performance of RAG retrievals using the Tonic Validate framework:

1. Add the following keys to your .env file:
  TONIC_VALIDATE_API_KEY=your_tonic_validate_api_key
  TONIC_VALIDATE_PRODUCT_KEY=your_tonic_validate_product_key
  TONIC_VALIDATE_BENCHMARK_KEY=your_tonic_validate_benchmark_key  # Optional
2. Run the evaluation script.
   python evaluate.py

## Troubleshooting

- **Issue**: Docker container fails to start
- **Solution**: Ensure Docker is running and you have sufficient system resources
- **Issue**: "Module not found" error when running locally
- **Solution**: Verify that you've activated the correct Conda environment
- **Issue**: API calls failing
- **Solution**: Check that your OpenAI API key is correctly set in the .env file

For more issues, please check our FAQ or open an issue on GitHub.

### Contributing
We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add some AmazingFeature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

### License
