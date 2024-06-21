from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Specify the model name
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")