
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from datetime import datetime, timedelta

# Assuming analyzer.py is in the same directory
from analyzer import SentimentAnalyzer

# --- Configuration ---
COMPLIANCE_RISK_WORDS = [
    "fraud", "scandal", "investigation", "litigation", "breach",
    "non-compliance", "regulatory", "penalty", "fine", "sanction",
    "embezzlement", "misconduct", "corruption", "whistleblower",
    "subpoena", "audit", "disclosure", "insider trading", "antitrust"
]

# --- Helper Functions ---
def generate_dummy_data(num_entries=100):
    """
    Generates dummy financial text data with sentiment and dates for demonstration.
    In a real scenario, this would come from ingest.py and analyzer.py.
    """
    data = []
    start_date = datetime.now() - timedelta(days=num_entries)
    analyzer = SentimentAnalyzer() # Initialize analyzer

    sample_texts = [
        "The company announced strong quarterly results, exceeding expectations.",
        "Revenue fell short due to unexpected market downturns and supply chain issues.",
        "New product launch received positive reviews, boosting investor confidence.",
        "Facing a regulatory investigation regarding past financial disclosures.",
        "Partnership with a leading tech firm is expected to drive significant growth.",
        "A class-action lawsuit has been filed against the management for alleged fraud.",
        "Market conditions remain stable, with minor fluctuations in stock price.",
        "The CEO expressed optimism about future prospects despite current challenges.",
        "Concerns about data breach led to a temporary halt in operations.",
        "The company received a substantial government contract, ensuring long-term stability."
    ]

    for i in range(num_entries):
        date = start_date + timedelta(days=i)
        text = sample_texts[i % len(sample_texts)]
        sentiment_result = analyzer.analyze_sentiment(text)
        
        data.append({
            "date": date,
            "text": text,
            "sentiment_label": sentiment_result["label"],
            "sentiment_score": sentiment_result["score"]
        })
    return pd.DataFrame(data)

def flag_compliance_risks(text: str, risk_words: list[str]) -> list[str]:
    """
    Flags regulatory compliance risk words in a given text.
    Args:
        text (str): The input text.
        risk_words (list[str]): A list of words to flag as compliance risks.
    Returns:
        list[str]: A list of found risk words in the text.
    """
    found_risks = []
    # Create a regex pattern to find whole words, case-insensitive
    pattern = r'\b(' + '|'.join(re.escape(word) for word in risk_words) + r')\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    for match in matches:
        found_risks.append(match.lower()) # Store in lowercase for consistency
    return list(set(found_risks)) # Return unique risk words

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Alternative Data Intelligence Pipeline")

st.title("📈 Alternative Data Intelligence Pipeline Dashboard")
st.markdown("Analyze financial text data for sentiment trends and regulatory compliance risks.")

# Load or generate data
# In a real application, you would load data from a persistent store or directly from ingest/analyzer
@st.cache_data
def load_data():
    return generate_dummy_data(num_entries=365)

df = load_data()

# Apply compliance flagging
df["compliance_risks"] = df["text"].apply(lambda x: flag_compliance_risks(x, COMPLIANCE_RISK_WORDS))
df["has_risk"] = df["compliance_risks"].apply(lambda x: len(x) > 0)

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

# Date Range Slider
min_date = df["date"].min().date()
max_date = df["date"].max().date()
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_dates) == 2:
    start_date_filter = datetime.combine(selected_dates[0], datetime.min.time())
    end_date_filter = datetime.combine(selected_dates[1], datetime.max.time())
    df_filtered = df[(df["date"] >= start_date_filter) & (df["date"] <= end_date_filter)]
else:
    df_filtered = df.copy()

# Sentiment Filter
sentiment_options = df_filtered["sentiment_label"].unique().tolist()
selected_sentiments = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=sentiment_options,
    default=sentiment_options
)
df_filtered = df_filtered[df_filtered["sentiment_label"].isin(selected_sentiments)]

# Compliance Risk Filter
risk_filter_option = st.sidebar.radio(
    "Filter by Compliance Risk",
    ("Show All", "Show Only Risks", "Hide Risks"),
    index=0
)

if risk_filter_option == "Show Only Risks":
    df_filtered = df_filtered[df_filtered["has_risk"] == True]
elif risk_filter_option == "Hide Risks":
    df_filtered = df_filtered[df_filtered["has_risk"] == False]


st.subheader(f"Analysis for {len(df_filtered)} Entries")

# --- Sentiment Trend Visualization ---
st.subheader("Sentiment Trends Over Time")

# Aggregate sentiment for plotting
sentiment_over_time = df_filtered.groupby(df_filtered["date"].dt.date)["sentiment_score"].mean().reset_index()
sentiment_over_time.columns = ["Date", "Average Sentiment Score"]

fig_sentiment = px.line(
    sentiment_over_time,
    x="Date",
    y="Average Sentiment Score",
    title="Average Sentiment Score Daily Trend",
    labels={
        "Date": "Date",
        "Average Sentiment Score": "Average Sentiment Score (0-1)"
    },
    hover_data={
        "Date": "|%Y-%m-%d",
        "Average Sentiment Score": ":.2f"
    }
)
fig_sentiment.update_traces(mode='lines+markers')
st.plotly_chart(fig_sentiment, use_container_width=True)

# --- Compliance Risk Overview ---
st.subheader("Regulatory Compliance Risk Overview")

if not df_filtered.empty:
    risk_counts = df_filtered["has_risk"].value_counts().rename(index={True: "Has Risk", False: "No Risk"})
    fig_risk = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        title="Proportion of Entries with Compliance Risks",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("### Details of Entries with Compliance Risks")
    risk_entries = df_filtered[df_filtered["has_risk"]].copy()
    if not risk_entries.empty:
        # Display relevant columns for risk entries
        st.dataframe(risk_entries[["date", "text", "sentiment_label", "compliance_risks"]].sort_values(by="date", ascending=False))
    else:
        st.info("No entries with compliance risks found in the filtered data.")
else:
    st.info("No data available for the selected filters.")

# --- Raw Data Display ---
st.subheader("Raw Data (Filtered)")
st.dataframe(df_filtered.sort_values(by="date", ascending=False))

# ==============================================================================
# 🤖 UNIFIED SECURE DEEPSEEK RAG CHAT MODULE
# ==============================================================================
import os
import sys
from dotenv import load_dotenv

st.write("---")
st.header("🤖 Query Alternative Data Engine (DeepSeek RAG)")
st.markdown("Ask natural language questions to cross-examine your local document index securely via DeepSeek.")

# 1. Resolve paths dynamically relative to where dashboard.py sits
current_dir = os.path.dirname(os.path.abspath(__file__))

# Attempt to load from .env or .env.txt depending on how Windows formatted it
env_path_normal = os.path.join(current_dir, '.env')
env_path_txt = os.path.join(current_dir, '.env.txt')

if os.path.exists(env_path_txt):
    load_dotenv(dotenv_path=env_path_txt)
else:
    load_dotenv(dotenv_path=env_path_normal)

api_key_exists = os.getenv("DEEPSEEK_API_KEY")

if not api_key_exists:
    st.error("⚠️ **DEEPSEEK_API_KEY Missing!** Please ensure you have created a `.env` file containing your key in the project root to unlock this interactive workspace module.")
else:
    # 2. Key is found! Now let's cleanly inject the folder path and import the pipeline
    try:
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        import rag_pipeline
        
        # Detect which query engine function name Manus compiled under the hood
        if hasattr(rag_pipeline, "query_rag"):
            query_engine = rag_pipeline.query_rag
        elif hasattr(rag_pipeline, "query"):
            query_engine = rag_pipeline.query
        elif hasattr(rag_pipeline, "ask"):
            query_engine = rag_pipeline.ask
        else:
            query_engine = None
            
        if query_engine:
            # 3. Render the interactive user text box
            user_query = st.text_input("Enter your market research or audit question:")
            if user_query:
                with st.spinner("DeepSeek is analyzing your local database vectors..."):
                    try:
                        response = query_engine(user_query)
                        st.markdown("### 📋 DeepSeek Response Analysis")
                        st.info(response)
                    except Exception as e:
                        st.error(f"Execution Error inside your RAG pipeline file: {str(e)}")
        else:
            st.warning("⚠️ Connected to `rag_pipeline.py` successfully, but could not detect a main query function (like `query_rag`). Please check the contents of your pipeline script.")
            
    except ImportError as e:
        st.error(f"⚠️ **Dependency or Hook Missing during pipeline setup:** {str(e)}")