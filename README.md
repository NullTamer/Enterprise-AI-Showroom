# Enterprise AI Showroom: Financial Intelligence Pipeline

This repository showcases a professional, end-to-end MLOps workflow for deploying LLM-based solutions in high-stakes enterprise environments. The project is organized into a three-part sequential architectural narrative, demonstrating the progression from raw data to a production-ready, evaluated system.

## 🚀 Architectural Narrative

### 1. Secure Knowledge Grounding (Advanced RAG)
The foundation of the pipeline uses **Retrieval-Augmented Generation (RAG)** to ground LLM responses in specific, private financial datasets. 
- **Core Technology:** DeepSeek API, Vector Embeddings.
- **Goal:** Eliminate hallucinations by providing the model with relevant context before generation.

### 2. Behavioral Syntax Alignment (Fine-Tuning Sequel)
Moving beyond basic grounding, this module demonstrates **Behavioral Alignment** using **QLoRA (Quantized Low-Rank Adaptation)**.
- **Core Technology:** QLoRA Adapters, JSON Schema Validation.
- **Goal:** Ensure the model follows strict enterprise syntax requirements (e.g., JSON/SQL outputs) and aligns with specific corporate communication styles.

### 3. Enterprise MLOps Evaluation Dashboard (Quality Gate)
The final stage is a rigorous **Evaluation Quality Gate** that benchmarks the hybrid system against standard base models.
- **Core Technology:** RAGAS-inspired metrics (Faithfulness, Context Recall, Answer Relevance).
- **Goal:** Provide quantifiable proof of performance improvements, ensuring the system meets enterprise SLAs for accuracy and latency.

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit
- **AI/LLM:** DeepSeek, QLoRA
- **Data Handling:** Pandas, NumPy
- **Visualization:** Plotly
- **DevOps:** Git, Dotenv

## 📂 Project Structure
```text
Enterprise-AI-Showroom/
├── dashboard.py         # Main entry point (Streamlit UI)
├── eval_suite.py        # MLOps evaluation module
├── rag_pipeline.py      # Core RAG logic
├── analyzer.py          # Sentiment analysis module
├── ingest.py            # Data ingestion scripts
├── .env.example         # Template for environment variables
├── .gitignore           # Git exclusion rules
└── README.md            # Project documentation
```

## ⚙️ Setup Instructions
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file based on `.env.example` and add your `DEEPSEEK_API_KEY`.
4. Run the dashboard: `streamlit run dashboard.py`
