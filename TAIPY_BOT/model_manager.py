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
            return "I apologize, but the model is not properly initialized."

        try:
            if prompt is None or len(prompt.strip()) == 0:
                logger.error("Empty or None prompt received")
                return "I apologize, but I received an empty message. Could you please provide some input?"

            # Add a prefix to guide the model towards specific topics
            topic_prefix = "Answer the following question about technology, news, or foreign affairs: "
            full_prompt = topic_prefix + prompt

            inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
            attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=400,
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
                return "I apologize, but I couldn't generate a response. Please try again."
            
            response = generated_text[len(full_prompt):].strip()
            
            if len(response) == 0:
                logger.warning("Generated response is empty")
                return "I apologize, but I couldn't generate a meaningful response. Could you please rephrase your input?"
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I'm having trouble generating a response. Error: {str(e)}"

    def cleanup(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Initialize global model manager
model_manager = ModelManager()

# Add these variables to make them available globally
conversation = {"Conversation": ["Who are you?", "Hello! I'm an AI assistant specialized in technology, news, and foreign affairs. How can I help you today?"]}
current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
is_generating = False
