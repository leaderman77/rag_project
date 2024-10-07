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

- **Implements multiple retrieval methods**:
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
   ```markdown
   git clone https://github.com/your-username/smartretrieval-rag.git
   cd smartretrieval-rag

3. Provide your OpenAI API Key in .env file located in the root directory:
   ```markdown
   OPENAI_API_KEY=your_openai_api_key_here

4. Build and run the Docker container:
   ```markdown
   docker-compose up --build

### Option 2: Using Conda Environment

1. Clone the repository:
   ```markdown
   git clone https://github.com/your-username/smartretrieval-rag.git
   cd smartretrieval-rag
3. Create and activate the Conda environment:
   ```markdown
   conda env create -f environment.yml
   conda activate rag_project_env
4. Create a .env file in the root directory and add your OpenAI API Key:
   ```markdown
   OPENAI_API_KEY=your_openai_api_key_here
5. Run the application:
   ```markdown
   python main.py

### Access to Gradio and FastAPI
- Access the Gradio interface at [http://localhost:8000](http://localhost:8000/) in your web browser.
- To use the FastAPI service, send a POST request to http://localhost:8000/ask as described in Option 1.

## Project Structure

```plaintext
smartretrieval-rag/
│
├── main.py                                # Entry point for the application
├── data/                                  # AI and LLM research papers
├── eval_questions/
│   ├── benchmark.json                     # Benchmark questions and ground truth used for evaluation
├── flagged/                               # Flagged data directory
├── resources/
│   └── text_qa_template.txt               # Response template
├── storage/
│   ├── auto_index/                        # Auto-Merging retrieval based vector indexing
│   ├── base_index/                        # Conventional retrieval based vector indexing
│   ├── kg_index/                          # Knowledge-graph retrieval based knowledge graph indexing
│   └── sentence_index/                    # Sentence window retrieval based vector indexing
├── Dockerfile                             # Docker configuration
├── docker-compose.yml                     # Docker Compose configuration
├── environment.yml                        # Conda environment specification
└── requirements.txt                       # Python dependencies, used for docker
├── .env                                   # Environment variables file
├── .gitignore      
├── docker-compose.yml
├── Dockerfile
├── environment.yml                        # Conda environment specification
├── evaluation.py                          # Evaluation file
├── evaluation_results_experiement_1.xlsx
├── evaluation_results_experiement_2.xlsx
├── final_results_test.xlsx              
├── load_papers.py                         # Load dataset papers
├── models.py                              # FastAPI model
├── parsed_documents.pkl                   # Stored dataset
├── process_documents.py                   # Process documents
├── process_retriever_index.py             # Creating and loading retrievals
├── rag_engine.py                          # RAG engines called
├── rag_result_analysis.ipynb              # Evaluation analysis
├── README.md
├── requirements.txt                       # Python dependencies
├── statistical_analysis.py                # Statistical analysis of evaluation
├── statistical_analysis_results.xlsx      # Statistical analysis of evaluation
└── utils.py                               # Utilty methods
```

### Evaluation
To evaluate the performance of RAG retrievals using the Tonic Validate framework:

1. Add the following keys to your .env file:
```markdown
TONIC_VALIDATE_API_KEY=your_tonic_validate_api_key
TONIC_VALIDATE_PRODUCT_KEY=your_tonic_validate_product_key
TONIC_VALIDATE_BENCHMARK_KEY=your_tonic_validate_benchmark_key  # Optional

2. Run the evaluation script.
```markdown
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
