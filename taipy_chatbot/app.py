import taipy as tp
from taipy.gui import Gui
from chatbot.chatbot_logic import get_chatbot_response

# Initialize the chatbot conversation
conversation = []

# Function to handle user input
def send_message():
    global message, conversation_display
    print(f"send_message triggered with message: '{message}'")  # Debug to check input value
    if message.strip() != "":  # Check if the message is not empty
        conversation.append(f"User: {message}")
        bot_reply = get_chatbot_response(message)  # Get bot reply
        conversation.append(f"Bot: {bot_reply}")
        print(f"Bot reply: {bot_reply}")  # Debug bot's reply
        message = ""  # Clear the message after sending
    else:
        print("Empty message, not processing.")
    update_conversation()  # Update the conversation display

# Update conversation display
def update_conversation():
    global conversation_display
    conversation_display = "\n\n".join(conversation)
    print(f"Updated conversation display: {conversation_display}")  # Debug conversation display update

# Variables to bind to the GUI
conversation_display = ""  # For displaying the chat
message = ""  # For user input

# Define the Taipy interface layout
page = """
<|layout|columns=1|gap=10px|>
    <|conversation_display|text|bind=conversation_display|style="height: 400px; overflow-y: scroll; border: 1px solid black; padding: 10px; font-family: Arial, sans-serif; background-color: #f9f9f9;"|>
    <|message|input|bind=message|on_submit=send_message|label=Type your message here...|>
    <|Send|button|on_action=send_message|style="margin-top: 10px;"|>
<|layout|>
"""

# Create a GUI instance
gui = Gui(page)

# Start the Taipy GUI server
if __name__ == "__main__":
    update_conversation()  # Initialize the conversation display
    gui.run()
