import os
import sys
import logging
from typing import Optional, Dict, List, Tuple
from taipy.gui import Gui, State, notify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
import socket
import requests

# Set custom directories
os.environ['TMPDIR'] = 'D:/custom_temp'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'
os.environ['TORCH_HOME'] = 'D:/pytorch_cache'

warnings.filterwarnings("ignore", category=ResourceWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_CONTEXT_LENGTH = 150
MAX_CONVERSATION_HISTORY = 50
MAX_MESSAGE_LENGTH = 500
MODEL_NAME = "distilgpt2"
NEWS_API_KEY = "1e9cd35b69324a788819d59b5995cec8"  # Replace with your actual News API key

# Initialize global state variables
conversation = {
    "Conversation": ["Who are you?", "Hi! I am GPT-2. How can I help you today?"]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
is_generating = False
context = (
    "You are a knowledgeable and friendly AI assistant with a human-like personality. "
    "You're well-informed about current events, technology, and various topics. "
    "You engage in natural conversations, provide detailed explanations, and can discuss "
    "recent news and developments. Always strive to be helpful, empathetic, and accurate.\n\n"
)

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.initialize_model()

    def initialize_model(self) -> None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Initializing model on device: {device}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def generate_response(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,  # Limit the response length
                do_sample=True,
                top_p=0.9,  # Adjusted for better quality
                top_k=40,  # Adjusted for better quality
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I'm having trouble generating a response. Error: {str(e)}"

    def cleanup(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Initialize global model manager
model_manager = ModelManager()

def validate_message(message: str) -> None:
    if not message.strip():
        raise ValueError("Message cannot be empty")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters")

def fetch_recent_news():
    try:
        url = f"https://newsapi.org/v2/top-headlines?apiKey={NEWS_API_KEY}&q=cricket"
        response = requests.get(url)
        news = response.json()
        if news["status"] == "ok":
            return [article["title"] for article in news["articles"][:5]]
        else:
            return []
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

def update_context(state: State) -> str:
    global context
    validate_message(state.current_user_message)
    
    # Fetch recent news
    recent_news = fetch_recent_news()
    news_context = "\n".join([f"- {news}" for news in recent_news])
    
    # Update context with recent news and user's message
    context += f"\n\nRecent news:\n{news_context}\n\n"
    context += f"Human: {state.current_user_message}\nAI:"
    
    # Trim context if too long
    if len(context) > MAX_CONTEXT_LENGTH:
        context_parts = context.split('\n')
        context = '\n'.join(context_parts[-30:])  # Keep more context for better coherence
    
    return model_manager.generate_response(context)

def on_init(state: State) -> None:
    """Initialize the state when the application starts."""
    logger.info("Initializing chat application")
    
    # Initialize state with global variables
    state.conversation = conversation
    state.current_user_message = current_user_message
    state.past_conversations = past_conversations
    state.selected_conv = selected_conv
    state.selected_row = selected_row
    state.is_generating = is_generating
    
    logger.info("Chat application initialized successfully")

def send_message(state: State) -> None:
    """Handle sending a message and getting a response."""
    global conversation
    
    if state.is_generating:
        notify(state, "warning", "Please wait for the current response to complete")
        return

    try:
        # Ensure the current user message is defined
        current_user_message = state.current_user_message.strip()
        validate_message(current_user_message)  # Validate the trimmed message
        
        state.is_generating = True
        notify(state, "info", "Generating response...")
        
        answer = update_context(state)  # Generate the AI's response
        
        # Update conversation with limits
        conv = state.conversation.copy()
        conv["Conversation"].extend([current_user_message, answer])
        if len(conv["Conversation"]) > MAX_CONVERSATION_HISTORY:
            conv["Conversation"] = conv["Conversation"][-MAX_CONVERSATION_HISTORY:]
        
        state.current_user_message = ""  # Clear the input field
        state.conversation = conv
        conversation = conv  # Update global state
        state.selected_row = [len(conv["Conversation"])]
        
        notify(state, "success", "Response received!")
        
    except ValueError as ve:
        notify(state, "warning", str(ve))
    except Exception as ex:
        logger.error(f"Error in send_message: {ex}")
        notify(state, "error", "An error occurred while processing your message")
    finally:
        state.is_generating = False

def reset_chat(state: State) -> None:
    """Reset the chat to its initial state."""
    global conversation, context, past_conversations
    
    if state.conversation["Conversation"]:
        past_conversations.append([
            len(past_conversations),
            state.conversation.copy()
        ])
        state.past_conversations = past_conversations
    
    # Reset to initial state
    context = (
        "You are a knowledgeable and friendly AI assistant with a human-like personality. "
        "You're well-informed about current events, technology, and various topics. "
        "You engage in natural conversations, provide detailed explanations, and can discuss "
        "recent news and developments. Always strive to be helpful, empathetic, and accurate.\n\n"
    )
    
    conversation = {
        "Conversation": ["Who are you?", "Hi! I am GPT-2. How can I help you today?"]
    }
    
    state.conversation = conversation
    state.current_user_message = ""
    state.selected_row = [1]
    state.is_generating = False

def tree_adapter(item: list) -> Tuple[str, str]:
    """Adapt conversation items for tree display."""
    identifier = str(item[0])
    if len(item[1]["Conversation"]) > 3:
        return (identifier, item[1]["Conversation"][2][:50] + "...")
    return (identifier, "Empty conversation")

def select_conv(state: State, var_name: str, value) -> None:
    """Handle selection of a past conversation."""
    global conversation, context
    
    if not value or not value[0]:
        return
        
    conv_idx = value[0][0]
    if conv_idx >= len(state.past_conversations):
        notify(state, "error", "Invalid conversation selected")
        return
        
    selected_conv = state.past_conversations[conv_idx][1]
    state.conversation = selected_conv.copy()
    conversation = selected_conv.copy()
    
    # Rebuild context
    context = (
        "You are a knowledgeable and friendly AI assistant with a human-like personality. "
        "You're well-informed about current events, technology, and various topics. "
        "You engage in natural conversations, provide detailed explanations, and can discuss "
        "recent news and developments. Always strive to be helpful, empathetic, and accurate.\n\n"
    )
    
    for i in range(2, len(selected_conv["Conversation"]), 2):
        context += (f"\nHuman: {selected_conv['Conversation'][i]}\n"
                   f"AI: {selected_conv['Conversation'][i + 1]}")
    
    state.selected_row = [len(selected_conv["Conversation"])]

# GUI Layout
page = """
<|layout|columns=300px 1|
<|part|class_name=sidebar|
# Taipy **Chat**{: .color-primary}
<|New Conversation|button|class_name=fullwidth plain|on_action=reset_chat|>
### Previous activities
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|>

<|part|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|>
<|"Generating..."|text|class_name=generating-text|visible={is_generating}|>
|>
|>
|>
"""

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

if __name__ == "__main__":
    try:
        # Test the model
        test_prompt = "Hello, how are you?"
        logger.info(f"Testing model with prompt: {test_prompt}")
        response = model_manager.generate_response(test_prompt)
        logger.info(f"Model test response: {response}")

        # Initialize Gui without on_init parameter
        gui = Gui(page)
        
        # Add on_init function separately
        gui.on_init = on_init
        
        port = find_free_port()
        print(f"Attempting to start application on port {port}")
        gui.run(
            dark_mode=True,
            title="💬 Taipy Chat",
            port=port,
            debug=True  # Set to True for more detailed error messages
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        sys.exit(1)
    finally:
        model_manager.cleanup()
        print("Application closed.")
