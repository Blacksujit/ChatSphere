import logging
from typing import Tuple
from taipy.gui import State, notify
from model_manager import ModelManager
import requests

model_manager = ModelManager()

logger = logging.getLogger(__name__)

MAX_CONTEXT_LENGTH = 2000
MAX_CONVERSATION_HISTORY = 50
MAX_MESSAGE_LENGTH = 50
NEWS_API_KEY = "your_news_api_key"

# Initialize global state variables
conversation = {
    "Conversation": ["Who are you?", "Hello! I'm an AI assistant specialized in technology, news, and foreign affairs. How can I help you today?"]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
is_generating = False
context = (
    "You are a knowledgeable and friendly AI assistant specialized in technology, news, and foreign affairs. "
    "You're well-informed about current events, technological advancements, and global politics. "
    "You engage in natural conversations, provide detailed explanations, and can discuss "
    "recent developments in these fields. Always strive to be helpful, empathetic, and accurate. "
    "Provide coherent and contextual responses. Keep your responses concise and informative.\n\n"
)

def validate_message(message: str) -> None:
    if not message.strip():
        raise ValueError("Message cannot be empty")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters")

def fetch_recent_news():
    try:
        url = f"https://newsapi.org/v2/top-headlines?apiKey={NEWS_API_KEY}&category=technology,science,business"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        news = response.json()
        if news["status"] == "ok":
            return [article["title"] for article in news["articles"][:5]]
        else:
            logger.warning(f"News API returned non-OK status: {news['status']}")
            return []
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []

def update_context(state: State) -> str:
    global context
    validate_message(state.current_user_message)
    
    recent_news = fetch_recent_news()
    news_context = "\n".join([f"- {news}" for news in recent_news])
    
    context += f"\n\nRecent tech and global news:\n{news_context}\n\n"
    context += f"Human: {state.current_user_message}\nAI:"
    
    if len(context) > MAX_CONTEXT_LENGTH:
        context_parts = context.split('\n')
        context = '\n'.join(context_parts[-50:])
    
    response = model_manager.generate_response(context)
    
    # Post-process the response to ensure relevance to tech, news, or foreign affairs
    response_lines = response.split('\n')
    response = response_lines[0] if response_lines else ""  # Take only the first line
    response = response.strip()
    
    if not response or len(response) < 20 or response.lower() == state.current_user_message.lower():
        response = "I apologize, but I couldn't generate a relevant response about technology, news, or foreign affairs. Could you please provide more context or rephrase your question?"
    
    return response

def on_init(state: State) -> None:
    logger.info("Initializing chat application")
    
    state.conversation = conversation.copy()
    state.current_user_message = current_user_message
    state.past_conversations = past_conversations.copy()
    state.selected_conv = selected_conv
    state.selected_row = selected_row.copy()
    state.is_generating = is_generating
    
    logger.info("Chat application initialized successfully")

def send_message(state: State) -> None:
    global conversation
    
    if state.is_generating:
        notify(state, "warning", "Please wait for the current response to complete")
        return

    try:
        current_user_message = state.current_user_message.strip()
        validate_message(current_user_message)
        
        state.is_generating = True
        notify(state, "info", "Generating response about technology, news, or foreign affairs...")
        
        answer = update_context(state)
        
        conv = state.conversation.copy()
        conv["Conversation"].extend([current_user_message, answer])
        if len(conv["Conversation"]) > MAX_CONVERSATION_HISTORY:
            conv["Conversation"] = conv["Conversation"][-MAX_CONVERSATION_HISTORY:]
        
        state.current_user_message = ""
        state.conversation = conv
        conversation = conv
        state.selected_row = [len(conv["Conversation"])]
        
        notify(state, "success", "Response received!")
        
    except ValueError as ve:
        notify(state, "warning", str(ve))
    except Exception as ex:
        logger.error(f"Error in send_message: {ex}")
        notify(state, "error", "An error occurred while processing your message about technology, news, or foreign affairs")
    finally:
        state.is_generating = False

def reset_chat(state: State) -> None:
    global conversation, context, past_conversations
    
    if state.conversation and state.conversation.get("Conversation"):
        past_conversations.append([
            len(past_conversations),
            state.conversation.copy()
        ])
        state.past_conversations = past_conversations.copy()
    
    context = (
        "You are a knowledgeable and friendly AI assistant specialized in technology, news, and foreign affairs. "
        "You're well-informed about current events, technological advancements, and global politics. "
        "You engage in natural conversations, provide detailed explanations, and can discuss "
        "recent developments in these fields. Always strive to be helpful, empathetic, and accurate. "
        "Provide coherent and contextual responses. Keep your responses concise and informative.\n\n"
    )
    
    conversation = {
        "Conversation": ["Who are you?", "Hello! I'm an AI assistant specialized in technology, news, and foreign affairs. How can I help you today?"]
    }
    
    state.conversation = conversation.copy()
    state.current_user_message = ""
    state.selected_row = [1]
    state.is_generating = False

def tree_adapter(item: list) -> Tuple[str, str]:
    identifier = str(item[0])
    if item[1] and "Conversation" in item[1] and len(item[1]["Conversation"]) > 3:
        return (identifier, item[1]["Conversation"][2][:50] + "...")
    return (identifier, "Empty conversation")

def select_conv(state: State, var_name: str, value) -> None:
    global conversation, context
    
    if not value or not value[0]:
        return
        
    try:
        conv_idx = int(value[0])
        if conv_idx < 0 or conv_idx >= len(state.past_conversations):
            raise ValueError("Invalid conversation index")
    except ValueError:
        notify(state, "error", "Invalid conversation selected")
        return
        
    selected_conv = state.past_conversations[conv_idx][1]
    state.conversation = selected_conv.copy()
    conversation = selected_conv.copy()
    
    context = (
        "You are a knowledgeable and friendly AI assistant specialized in technology, news, and foreign affairs. "
        "You're well-informed about current events, technological advancements, and global politics. "
        "You engage in natural conversations, provide detailed explanations, and can discuss "
        "recent developments in these fields. Always strive to be helpful, empathetic, and accurate. "
        "Provide coherent and contextual responses. Keep your responses concise and informative.\n\n"
    )
    
    if "Conversation" in selected_conv:
        for i in range(2, len(selected_conv["Conversation"]), 2):
            context += (f"\nHuman: {selected_conv['Conversation'][i]}\n"
                       f"AI: {selected_conv['Conversation'][i + 1]}")
    
    state.selected_row = [len(selected_conv.get("Conversation", []))]
