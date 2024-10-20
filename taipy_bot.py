# # !pip install taipy
# # !pip install service_identity

# import os
# import sys
# import time
# from taipy.gui import Gui, State, notify
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch


# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="openai-community/gpt2")


# # Set up conversation context
# conversation = {
#     "Conversation": ["Who are you?", "Hi! I am GPT-2. How can I help you today?"]
# }
# current_user_message = ""
# past_conversations = []
# selected_conv = None
# selected_row = [1]

# def on_init(state: State) -> None:
#     state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?"
#     state.conversation = {
#         "Conversation": ["Who are you?", "Hi! I am GPT-2. How can I help you today?"]
#     }
#     state.current_user_message = ""
#     state.past_conversations = []
#     state.selected_conv = None
#     state.selected_row = [1]

# def generate_response(state: State, prompt: str) -> str:
#     try:
#         inputs = pipe.encode(prompt, return_tensors="pt")
#         outputs = pipe.generate(inputs, max_length=150, do_sample=True, top_p=0.95, top_k=60)
#         response = pipe.decode(outputs[0], skip_special_tokens=True)
#         return response.split(prompt)[-1].strip()
#     except Exception as e:
#         notify(state, "error", f"Error generating response: {str(e)}")
#         return "Error: Unable to generate response."

# def update_context(state: State) -> str:
#     state.context += f"\nHuman: {state.current_user_message}\nAI:"
#     answer = generate_response(state, state.context).replace("\n", "")
#     state.context += answer
#     state.selected_row = [len(state.conversation["Conversation"]) + 1]
#     return answer

# def send_message(state: State) -> None:
#     notify(state, "info", f"Sending message: {state.current_user_message}")  # Debugging output
#     try:
#         answer = update_context(state)
#         conv = state.conversation.copy()
#         conv["Conversation"].extend([state.current_user_message, answer])
#         state.current_user_message = ""  # Clear the message after sending
#         state.conversation = conv
#         notify(state, "success", "Response received!")
#     except Exception as ex:
#         on_exception(state, "send_message", ex)  # Call the exception handler

# def style_conv(state: State, idx: int, row: int) -> str:
#     if idx is None:
#         return None
#     elif idx % 2 == 0:
#         return "user_message"
#     else:
#         return "gpt_message"

# def on_exception(state, function_name: str, ex: Exception) -> None:
#     notify(state, "error", f"An error occurred in {function_name}: {ex}")

# def reset_chat(state: State) -> None:
#     state.past_conversations = state.past_conversations + [
#         [len(state.past_conversations), state.conversation]
#     ]
#     state.conversation = {
#         "Conversation": ["Who are you?", "Hi! I am GPT-2. How can I help you today?"]
#     }

# def tree_adapter(item: list) -> [str, str]:
#     identifier = str(item[0])  # Convert identifier to a string
#     if len(item[1]["Conversation"]) > 3:
#         return (identifier, item[1]["Conversation"][2][:50] + "...")
#     return (identifier, "Empty conversation")

# def select_conv(state: State, var_name: str, value) -> None:
#     state.conversation = state.past_conversations[value[0][0]][1]
#     state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?"
#     for i in range(2, len(state.conversation["Conversation"]), 2):
#         state.context += f"Human: \n {state.conversation['Conversation'][i]}\n\n AI:"
#         state.context += state.conversation["Conversation"][i + 1]
#     state.selected_row = [len(state.conversation["Conversation"]) + 1]

# past_prompts = []

# page = """
# <|layout|columns=300px 1|
# <|part|class_name=sidebar| 
# # Taipy **Chat**{: .color-primary} # {: .logo-text}
# <|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
# ### Previous activities ### {: .h5 .mt2 .mb-half}
# <|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
# |>

# <|part|class_name=p2 align-item-bottom table| 
# <|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>
# <|part|class_name=card mt1| 
# <|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
# |>
# |>
# |>
# """
# if __name__ == "__main__":
#     Gui(page).run(
#         debug=True, 
#         dark_mode=True, 
#         title="💬 Taipy Chat", 
#         port=5001,  # Use a different port here
#         allow_unsafe_werkzeug=True
#     )


# !pip install taipy
# !pip install service_identity

import os
import sys
import time
from taipy.gui import Gui, State, notify
from transformers import pipeline
import torch

# Load the GPT-2 model using transformers' pipeline
try:
    pipe = pipeline("text-generation", model="openai-community/gpt2")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

def on_init(state: State) -> None:
    print("Initializing the chat application.")
    
    # Initialize context and conversation
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?"
    state.conversation = {"Conversation": ["Who are you?", "Hi! I am GPT-2. How can I help you today?"]}
    state.current_user_message = ""
    state.past_conversations = []
    state.selected_conv = None
    state.selected_row = [1]

    print(f"Initialized context: {state.context}")

def generate_response(state: State, prompt: str) -> str:
    try:
        # Generate response from the model
        response = pipe(prompt, max_length=150, do_sample=True, top_p=0.95, top_k=60)
        return response[0]['generated_text'].split(prompt)[-1].strip()
    except Exception as e:
        notify(state, "error", f"Error generating response: {str(e)}")
        return "Error: Unable to generate response."

def update_context(state: State) -> str:
    state.context += f"\nHuman: {state.current_user_message}\nAI:"
    answer = generate_response(state, state.context).replace("\n", "")
    state.context += answer
    state.selected_row = [len(state.conversation["Conversation"]) + 1]
    return answer

def send_message(state: State) -> None:
    notify(state, "info", f"Sending message: {state.current_user_message}")
    try:
        answer = update_context(state)
        conv = state.conversation.copy()
        conv["Conversation"].extend([state.current_user_message, answer])
        state.current_user_message = ""
        state.conversation = conv
        notify(state, "success", "Response received!")
    except Exception as ex:
        on_exception(state, "send_message", ex)

def on_exception(state, function_name: str, ex: Exception) -> None:
    notify(state, "error", f"An error occurred in {function_name}: {str(ex)}")

def reset_chat(state: State) -> None:
    state.past_conversations.append([len(state.past_conversations), state.conversation])
    state.conversation = {"Conversation": ["Who are you?", "Hi! I am GPT-2. How can I help you today?"]}

def tree_adapter(item: list) -> (str, str):
    identifier = str(item[0])
    if len(item[1]["Conversation"]) > 3:
        return (identifier, item[1]["Conversation"][2][:50] + "...")
    return (identifier, "Empty conversation")

def select_conv(state: State, var_name: str, value) -> None:
    state.conversation = state.past_conversations[value[0][0]][1]
    # Rebuild the context based on selected conversation
    state.context = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?"
    for i in range(2, len(state.conversation["Conversation"]), 2):
        state.context += f"\nHuman: {state.conversation['Conversation'][i]}\nAI:"
        state.context += state.conversation["Conversation"][i + 1]
    state.selected_row = [len(state.conversation["Conversation"]) + 1]

# GUI Layout
page = """
<|layout|columns=300px 1| 
<|part|class_name=sidebar| 
# Taipy **Chat**{: .color-primary} # {: .logo-text}
<|New Conversation|button|class_name=fullwidth plain|id=reset_app_button|on_action=reset_chat|>
### Previous activities ### {: .h5 .mt2 .mb-half}
<|{selected_conv}|tree|lov={past_conversations}|class_name=past_prompts_list|multiple|adapter=tree_adapter|on_change=select_conv|>
|> 

<|part|class_name=p2 align-item-bottom table| 
<|{conversation}|table|style=style_conv|show_all|selected={selected_row}|rebuild|>
<|part|class_name=card mt1| 
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|change_delay=-1|>
|>
|>
|>
"""

if __name__ == "__main__":
    Gui(page).run(
        debug=True, 
        dark_mode=True, 
        title="💬 Taipy Chat", 
        port="auto",  # Let Taipy choose an available port
        allow_unsafe_werkzeug=True
    )


# simple transformer chatbot 
