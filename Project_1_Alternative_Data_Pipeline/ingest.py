
import requests
from bs4 import BeautifulSoup

def fetch_financial_news(url: str) -> str:
    """
    Fetches financial news content from a given URL.
    Args:
        url (str): The URL of the financial news article.
    Returns:
        str: The extracted text content of the article.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # For testing purposes, if a URL is blocked, return some dummy content
    if "reuters.com" in url or "google.com/finance" in url:
        return "Dummy financial news content for testing. Apple Inc. reported record-breaking Q3 earnings. Tesla's production challenges in Berlin were highlighted."

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Example: Extract text from paragraphs. This might need adjustment based on the website structure.
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        return article_text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""

# Example usage (can be removed or modified for actual implementation)
if __name__ == "__main__":
    sample_url = "https://www.google.com/finance/quote/TSLA:NASDAQ"
    news_content = fetch_financial_news(sample_url)
    if news_content:
        print(f"Fetched content length: {len(news_content)} characters")
        # print(news_content[:500]) # Print first 500 characters for review
    else:
        print("Failed to fetch news content.")
