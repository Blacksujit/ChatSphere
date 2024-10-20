from chatbot.nlp_processor import process_message

# Example predefined responses
responses = {
    "hi": "Hello! How can I help you?",
    "bye": "Goodbye! Have a great day!",
    "help": "Sure! I can help you with general information.",
    # Add more responses as needed
}

# Chatbot response logic
def get_chatbot_response(user_input):
    """Generates a response based on the user input."""
    try:
        processed_message = process_message(user_input)
        print(f"Processed message: '{processed_message}'")  # Debug output
        
        # Return a predefined response or a fallback
        return responses.get(processed_message, "I'm sorry, I don't understand that. Can you rephrase?")
    except Exception as e:
        return f"An error occurred: {str(e)}"
