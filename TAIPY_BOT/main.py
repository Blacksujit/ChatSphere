import logging
import sys
import os
from taipy.gui import Gui, State
from chat_logic import on_init, send_message, reset_chat, select_conv, conversation, current_user_message, past_conversations, selected_conv, selected_row, is_generating
from gui_layout import page
from utils import find_free_port
from model_manager import ModelManager

# Set custom directories
os.environ['TMPDIR'] = 'D:/custom_temp'
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache'
os.environ['TORCH_HOME'] = 'D:/pytorch_cache'

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

# Initialize ModelManager
model_manager = ModelManager()

if __name__ == "__main__":
    try:
        # Test the model
        test_prompt = "Hello, how are you?"
        logger.info(f"Testing model with prompt: {test_prompt}")
        response = model_manager.generate_response(test_prompt)
        logger.info(f"Model test response: {response}")

        # Initialize Gui with the page layout and required variables
        gui = Gui(page)
        
        # Add on_init function separately
        gui.on_init = on_init
        
        # Find an available port
        port = find_free_port()
        logger.info(f"Attempting to start application on port {port}")
        
        # Run the GUI with the required variables
        gui.run(
            dark_mode=True,
            title="💬 Taipy Chat",
            port=port,
            debug=True,  # Set to True for more detailed error messages
            data={
                "conversation": conversation,
                "current_user_message": current_user_message,
                "past_conversations": past_conversations,
                "selected_conv": selected_conv,
                "selected_row": selected_row,
                "is_generating": is_generating
            }
        )
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        sys.exit(1)
    finally:
        model_manager.cleanup()
        logger.info("Application closed.")
