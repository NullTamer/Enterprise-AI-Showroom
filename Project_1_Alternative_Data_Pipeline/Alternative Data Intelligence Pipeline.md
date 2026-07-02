
# Alternative Data Intelligence Pipeline

This project implements an **Alternative Data Intelligence Pipeline** designed to ingest unstructured financial text data, analyze its sentiment, provide a Retrieval-Augmented Generation (RAG) interface for querying, and visualize key insights through a Streamlit dashboard.

## Architecture Overview

The pipeline consists of the following modular Python scripts:

1.  **`ingest.py`**: Responsible for scraping or fetching unstructured financial text data from various sources, such as news articles or earnings call transcripts. It uses `requests` and `BeautifulSoup` for web scraping.
2.  **`analyzer.py`**: Ingests text data and performs sentiment analysis using a pre-trained model from the Hugging Face Transformers library. It provides sentiment labels (e.g., POSITIVE, NEGATIVE, NEUTRAL) and confidence scores.
3.  **`rag_pipeline.py`**: Utilizes LangChain to process financial text files. It chunks the text, generates embeddings using Hugging Face models, creates a FAISS vector store for efficient retrieval, and offers a Question-Answering (QA) interface powered by an LLM (e.g., OpenAI's GPT-3.5).
4.  **`dashboard.py`**: A Streamlit web application that serves as the frontend for the pipeline. It visualizes sentiment trends over time, displays a breakdown of compliance risks flagged by regex patterns, and allows users to interact with the analyzed data.

## Setup and Installation

To set up and run the project, follow these steps:

1.  **Clone the repository (if applicable) or navigate to the project directory:**
    ```bash
    cd Alternative-Data-Intelligence-Pipeline
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables (for `rag_pipeline.py`):**
    The `rag_pipeline.py` script uses `ChatOpenAI`, which requires an `OPENAI_API_KEY`. Set this environment variable before running the RAG pipeline:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
    *(Note: For production or persistent environments, consider using a more secure method for managing API keys.)*

## Running the Pipeline

### 1. Data Ingestion (Example)

To fetch a sample financial news article:

```bash
python3 ingest.py
```

This script will print the length of the fetched content. In a real scenario, you would integrate this to fetch multiple articles and save them for further processing.

### 2. Sentiment Analysis (Example)

To test the sentiment analyzer:

```bash
python3 analyzer.py
```

This will demonstrate sentiment analysis on a few sample texts.

### 3. RAG Pipeline (Example)

To create a vector store and test the QA interface:

```bash
python3 rag_pipeline.py
```

This will create a FAISS index (e.g., `./faiss_index_financial`) and answer sample queries. Ensure `OPENAI_API_KEY` is set.

### 4. Streamlit Dashboard

To launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`). The dashboard will display sentiment trends and compliance risk flags based on dummy data or your ingested and analyzed data.

## Project Structure

```
Alternative-Data-Intelligence-Pipeline/
├── ingest.py
├── analyzer.py
├── rag_pipeline.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## Dependencies

The `requirements.txt` file lists all necessary Python packages:

```
streamlit
transformers
langchain
langchain-community
langchain-openai
sentence-transformers
faiss-cpu
beautifulsoup4
pandas
matplotlib
plotly
requests
```

## Future Enhancements

*   Integration with real-time financial data APIs.
*   More sophisticated web scraping capabilities with error handling and dynamic content support.
*   Advanced NLP models for entity recognition and topic modeling.
*   Customizable compliance risk dictionaries and alerting systems.
*   User authentication and persistent data storage for the Streamlit dashboard.
