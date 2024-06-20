from transformers import pipeline, AutoTokenizer
from collections import Counter

# Initialize the sentiment analysis pipeline and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

special_tokens_count = len(tokenizer.build_inputs_with_special_tokens([]))

def chunk_text(text, chunk_size=512, overlap=50):
    """
    Splits the text into chunks of chunk_size with overlap to handle context better.
    Ensures that each chunk does not exceed the maximum token length.
    """
    tokens = text.split()
    chunks = []
    current_chunk = []

    adjusted_chunk_size = chunk_size - special_tokens_count

    for token in tokens:
        current_chunk.append(token)
        if len(tokenizer(" ".join(current_chunk))['input_ids']) > adjusted_chunk_size:
            # Remove the last token and save the current chunk
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            # Start a new chunk with overlap
            current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def analyze_sentiment(text):
    chunks = chunk_text(text)
    results = []
    for chunk in chunks:
        result = sentiment_pipeline(chunk)
        results.extend(result)
    
    return results

def map_star_to_sentiment(star):
    """
    Maps star rating to sentiment label.
    """
    star_rating = int(star.split()[0])  # Extract the star rating number
    if star_rating in [1, 2]:
        return "negative"
    elif star_rating == 3:
        return "neutral"
    elif star_rating in [4, 5]:
        return "positive"
    
def aggregate_results(results):
    """
    Aggregates the sentiment analysis results.
    """
    sentiment_scores = Counter()
    for result in results:
        sentiment_label = map_star_to_sentiment(result['label'])
        score = result['score']
        sentiment_scores[sentiment_label] += score

    total_scores = sum(sentiment_scores.values())
    aggregated_results = {label: (score / total_scores) for label, score in sentiment_scores.items()}
    
    return aggregated_results
