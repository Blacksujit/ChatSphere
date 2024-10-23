import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set custom directories
os.environ['TMPDIR'] = 'D:/custom_temp'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'
os.environ['TORCH_HOME'] = 'D:/pytorch_cache'

# Ensure the custom directories exist
for dir_path in [os.environ['TMPDIR'], os.environ['TRANSFORMERS_CACHE'], os.environ['TORCH_HOME']]:
    os.makedirs(dir_path, exist_ok=True)

logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/opt-125m"  # A small model that's not GPT-based

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_model()

    def initialize_model(self) -> None:
        try:
            logger.info(f"Initializing model on device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(self.device)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def generate_response(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer not initialized")
            return "Model initialization error. Please try again later."

        try:
            if not prompt or not prompt.strip():
                logger.error("Empty or None prompt received")
                return "Please provide a valid input."

            # Modify the prefix to be more flexible
            topic_prefix = "Provide a concise response about: "
            full_prompt = topic_prefix + prompt

            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=290,  # Reduced for more concise responses
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if generated_text is None:
                logger.error("Generated text is None")
                return "Unable to generate a response. Please try again."
            
            response = generated_text[len(full_prompt):].strip()
            
            if not response:
                logger.warning("Generated response is empty")
                return "I couldn't generate a relevant response. Could you rephrase your question?"
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"An error occurred: {str(e)}"

    def cleanup(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Initialize global model manager
model_manager = ModelManager()

# Add these variables to make them available globally
conversation = {
    "Conversation": [
        "Who are you?",
        "I'm an AI assistant specializing in technology, news, and foreign affairs. How can I help you?",
        "Hi",
        "Hello! What would you like to know about technology, news, or foreign affairs?",
        "What are today's updates on tech news?",
        "I'm sorry, but I don't have real-time information. Could you ask about a specific tech topic?",
        "What's the news of the day?",
        "I don't have current news. Can you ask about a specific news topic or event?",
        "What's today's news in cricket?",
        "I don't have today's cricket news. Could you ask about a specific cricket team or tournament?",
        "What is the current state of technology and news?",
        "I can't provide current information. What specific aspect of technology or news interests you?"
    ]
}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
is_generating = False
