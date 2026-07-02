import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Use the official DeepSeek integration module instead of OpenAI placeholders!
from langchain_deepseek import ChatDeepSeek

# Load environment variables
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path_normal = os.path.join(current_dir, '.env')
env_path_txt = os.path.join(current_dir, '.env.txt')

if os.path.exists(env_path_txt):
    load_dotenv(dotenv_path=env_path_txt)
else:
    load_dotenv(dotenv_path=env_path_normal)

def query_rag(user_question: str) -> str:
    """
    Main entry point function called by dashboard.py
    Initializes the FAISS database and queries DeepSeek.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY is missing from your environment configuration."

    # 2. Initialize the free local embedding engine (No key required for this part)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Create or Load your vector database index
    # (Creating a quick dummy index so your dashboard doesn't crash if it's empty)
    dummy_text_path = os.path.join(current_dir, "dummy_data.txt")
    if not os.path.exists(dummy_text_path):
        with open(dummy_text_path, "w") as f:
            f.write("Alternative Data Intelligence Pipeline Core: Market sentiment shows high compliance adherence across sector indices.")

    loader = TextLoader(dummy_text_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    db = FAISS.from_documents(docs, embeddings)
    
    # 4. Search local vector data for context matching the user's question
    search_results = db.similarity_search(user_question, k=2)
    context = "\n".join([doc.page_content for doc in search_results])
    
    # 5. Connect directly to DeepSeek's server using your key
    try:
        # Calls the ultra-fast, cheap deepseek-chat model
        llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=api_key,
            temperature=0.2
        )
        
        # Build a structured prompt blending your local data with the AI brain
        system_prompt = f"You are a financial intelligence analyst. Use the following piece of local context to answer the user's question accurately.\n\nContext:\n{context}\n\nQuestion: {user_question}"
        
        response = llm.invoke(system_prompt)
        return response.content
    except Exception as e:
        return f"DeepSeek API Communication Error: {str(e)}"