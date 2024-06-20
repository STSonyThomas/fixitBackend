from transformers import pipeline, AutoTokenizer
from collections import Counter
import firebase_admin
from firebase_admin import credentials, storage 
import os

cred = credentials.Certificate("./firebaseConfig.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'gs://sentimentanalyser-18331.appspot.com'
})

bucket = storage.bucket()

def download_model_from_firebase(blob_name, destination_file_name):
    """Downloads a file from Firebase storage."""
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {blob_name} downloaded to {destination_file_name}.")

model_dir = 'model'
# os.makedirs(model_dir, exist_ok=True)
# download_model_from_firebase('model/model.safetensors', os.path.join(model_dir, 'model.safetensors'))
# download_model_from_firebase('model/config.json', os.path.join(model_dir, 'config.json'))
# download_model_from_firebase('model/tokenizer_config.json', os.path.join(model_dir, 'tokenizer_config.json'))
# download_model_from_firebase('model/vocab.txt', os.path.join(model_dir, 'vocab.txt'))
# download_model_from_firebase('model/special_tokens_map.json', os.path.join(model_dir, 'special_tokens_map.json'))
# download_model_from_firebase('model/tokenizer.json', os.path.join(model_dir, 'tokenizer.json'))

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
