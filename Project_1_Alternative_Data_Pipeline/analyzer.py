
from transformers import pipeline

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis using a pre-trained Hugging Face Transformers model.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initializes the SentimentAnalyzer with a specified Hugging Face model.
        Args:
            model_name (str): The name of the pre-trained model to use for sentiment analysis.
                              Defaults to 'distilbert-base-uncased-finetuned-sst-2-english'.
        """
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyzes the sentiment of a given text.
        Args:
            text (str): The input text for sentiment analysis.
        Returns:
            dict: A dictionary containing the sentiment label (e.g., 'POSITIVE', 'NEGATIVE')
                  and the corresponding score.
        """
        if not text.strip():
            return {"label": "NEUTRAL", "score": 0.5} # Handle empty or whitespace-only text
        
        # Hugging Face pipeline returns a list of dictionaries, we take the first one
        result = self.sentiment_pipeline(text)[0]
        return result

    def analyze_batch_sentiment(self, texts: list[str]) -> list[dict]:
        """
        Analyzes the sentiment of a list of texts in a batch.
        Args:
            texts (list[str]): A list of input texts for sentiment analysis.
        Returns:
            list[dict]: A list of dictionaries, each containing the sentiment label and score
                        for the corresponding input text.
        """
        # Filter out empty strings to avoid errors with the pipeline
        processed_texts = [text for text in texts if text.strip()]
        if not processed_texts:
            return [{
                "label": "NEUTRAL",
                "score": 0.5
            }] * len(texts) # Return neutral for all if all are empty

        results = self.sentiment_pipeline(processed_texts)
        
        # Map results back to original texts, filling in neutral for empty ones
        full_results = []
        result_idx = 0
        for text in texts:
            if text.strip():
                full_results.append(results[result_idx])
                result_idx += 1
            else:
                full_results.append({"label": "NEUTRAL", "score": 0.5})
        return full_results

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    sample_text_positive = "The company reported excellent earnings and strong growth."
    sample_text_negative = "Despite efforts, the quarter showed significant losses."
    sample_text_neutral = "The stock opened at $150 today."
    sample_text_empty = ""

    print(f"Sentiment for \"{sample_text_positive}\": {analyzer.analyze_sentiment(sample_text_positive)}")
    print(f"Sentiment for \"{sample_text_negative}\": {analyzer.analyze_sentiment(sample_text_negative)}")
    print(f"Sentiment for \"{sample_text_neutral}\": {analyzer.analyze_sentiment(sample_text_neutral)}")
    print(f"Sentiment for \"{sample_text_empty}\": {analyzer.analyze_sentiment(sample_text_empty)}")

    batch_texts = [
        "The market is soaring, and investors are optimistic.",
        "Recession fears are growing, leading to market uncertainty.",
        "The company announced a new product line.",
        "",
        "Another positive development for the sector."
    ]
    print("\nBatch Sentiment Analysis:")
    for text, sentiment in zip(batch_texts, analyzer.analyze_batch_sentiment(batch_texts)):
        print(f"Text: \"{text}\" -> Sentiment: {sentiment}")
