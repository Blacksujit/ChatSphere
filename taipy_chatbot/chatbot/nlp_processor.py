import nltk
from nltk.tokenize import word_tokenize

# Ensure that necessary NLTK data is downloaded
nltk.download('punkt')

def process_message(message):
    """Tokenizes and normalizes the user input."""
    tokens = word_tokenize(message.lower())
    return tokens[0] if tokens else ""  # Return the first token or an empty string
