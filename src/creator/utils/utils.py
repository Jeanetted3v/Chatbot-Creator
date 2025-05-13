"""
Shared application state for the chatbot creator.
"""
from pathlib import Path
from hydra import compose, initialize_config_dir


def load_config():
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent.parent.parent
    config_dir = project_root / "config"

    with initialize_config_dir(
        config_dir=str(config_dir),
        job_name="Create Chatbot"
    ):
        cfg = compose(config_name="config")
    return cfg


state = {
    "mode": "",
    "summary": "",
    "creator": "",
    "prompt_creation_complete": False,
    "sim_prompt": "",
    "chat_prompt": "",
    "reasoning_prompt": "",
    "conversations": [],
    "current_conv_index": 0,
    "feedback_history": [],
    "reasoning_model": "gpt-4.1-mini",  # Default model name
    "response_model": "gpt-4.1-mini",  # Default model name
    "regeneration_count": 0
}

# Example structure for the feedback history:
# app_state["feedback_history"] = [
#     {
#         "conversation_index": 0,  # Index of the conversation in the conversations array 1 to 10 examples for each iteration
#         "conversation": [         # The actual conversation turns
#             {
#                 "customer_inquiry": "How can I reset my password?",
#                 "bot_response": "To reset your password, please go to the login page and click..."
#             },
#             # More turns...
#         ],
#         "feedback": "The bot response was helpful but could be more concise..."  # User feedback
#     },
#     # More feedback entries for other conversations...
# ]


def get(key, default=None):
    """Get a value from the application state."""
    return state.get(key, default)


def set(key, value):
    """Set a value in the application state."""
    state[key] = value
    return value


def update(data):
    """Update multiple values in the application state."""
    state.update(data)
    return state