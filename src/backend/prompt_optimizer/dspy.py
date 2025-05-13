"""To run:
python -m src.backend.prompt_optimizer.dspy

"""
import dspy
import os
import json
import glob
import logging
import yaml
from pathlib import Path
from typing import Optional
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dspy.teleprompt import *
from src.backend.utils.settings import SETTINGS

logger = logging.getLogger(__name__)


lm = dspy.LM('openai/gpt-4.1-mini', api_key=SETTINGS.OPENAI_API_KEY)
dspy.configure(lm=lm)


def load_conversations_from_json(directory_path="./data/convo"):
    examples = []
    json_files = glob.glob(f"{directory_path}/*.json")
    
    if not json_files:
        logger.info(f"No JSON files found in {directory_path}")
        return examples
        
    for file_path in json_files:
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                conversations = data if isinstance(data, list) else [data]
                
                for item in conversations:
                    user_question = item.get("customer_inquiry", "")
                    assistant_response = item.get("expected_output", "")
                    conversation_data = [
                        {"role": "user", "content": user_question}
                    ]
                    if assistant_response:
                        conversation_data.append({"role": "assistant", "content": assistant_response})
                    
                    example = dspy.Example(
                        user_question=user_question,
                        conversation=conversation_data
                    ).with_inputs("user_question", "conversation")
                    examples.append(example)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    logger.info(f"Successfully loaded {len(examples)} conversations from {directory_path}")
    return examples


def load_prompts_from_yaml(yaml_path="./config/query_handler_prompts.yaml"):
    try:
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            logger.error(f"Prompts file not found: {yaml_path}")
            return {}
            
        with open(yaml_path, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
            
        logger.info(f"Successfully loaded prompts from {yaml_path}")
        return prompts
    except Exception as e:
        logger.error(f"Error loading prompts from {yaml_path}: {e}")
        return {}


_prompts = load_prompts_from_yaml()


class ConversationContinuation(dspy.Signature):
    """
    Generates a response to continue a conversation based on user input and conversation history.
    
    Instructions:
    {instruction_prompt}
    
    Role:
    {role_prompt}
    """
    user_question: str = dspy.InputField()
    conversation: Optional[list] = dspy.InputField()
    response: str = dspy.OutputField()
    history: dict = dspy.OutputField()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Get the nested prompts from the YAML structure
        if _prompts and "query_handler_prompts" in _prompts:
            if "response_agent" in _prompts["query_handler_prompts"]:
                response_agent = _prompts["query_handler_prompts"]["response_agent"]
                instruction_prompt = response_agent.get("sys_prompt", "")
                role_prompt = response_agent.get("user_prompt", "")
            else:
                instruction_prompt = "No response agent prompts found in configuration."
                role_prompt = "Please check your configuration file."
        else:
            instruction_prompt = "No prompts found in configuration."
            role_prompt = "Please check your configuration file."
            
        # Format the docstring with the actual prompts
        self.__doc__ = self.__doc__.format(
            instruction_prompt=instruction_prompt,
            role_prompt=role_prompt
        )


class OpenAISemanticEvaluator:
    """Uses OpenAI embeddings for semantic similarity computation"""
    def __init__(self, model="text-embedding-3-small"):
        self.model = model
        self.client = openai.OpenAI(api_key=SETTINGS.OPENAI_API_KEY)
    
    def get_embedding(self, text):
        """Get OpenAI embedding for a text string"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding)
    
    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two texts"""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity


class GEval:
    """
    Compatible with DeepEval's ConversationalGEval metric implementation
    Uses input, actual_output, and expected_output for evaluation
    """
    def __init__(self, criteria, threshold=0.5, name="accuracy"):
        self.criteria = criteria
        self.threshold = threshold
        self.name = name
    
    def evaluate(self, example, prediction):
        conversation_history = ""
        for msg in example.conversation:
            role = msg["role"].capitalize()
            content = msg["content"]
            conversation_history += f"{role}: {content}\n"
        
        # Get the predicted response
        actual_output = prediction.response
        
        # Format the expected response (last assistant message)
        last_assistant_msgs = [msg["content"] for msg in example.conversation if msg["role"] == "assistant"]
        if not last_assistant_msgs:
            return 0.0
        expected_output = last_assistant_msgs[-1]
        
        # Use LLM to evaluate based on DeepEval's pattern
        evaluation_prompt = f"""
        {self.criteria}
        
        Conversation History:
        {conversation_history}
        
        User's Last Question: {example.user_question}
        Actual Output (Model Generated): {actual_output}
        Expected Output (Ground Truth): {expected_output}
        
        Consider the ENTIRE conversation context when evaluating accuracy.
        Rate the accuracy on a scale of 0.0 to 1.0, where:
        - 0.0 means completely inaccurate or incorrect
        - 1.0 means perfectly accurate and equivalent
        
        Output only a single float number between 0.0 and 1.0.
        """
        with dspy.context(lm=lm):
            result = dspy.Predict("prompt -> score")(prompt=evaluation_prompt)
            try:
                score = float(result.score)
                return min(max(score, 0.0), 1.0)  # Ensure score is between 0 and 1
            except ValueError:
                # Fallback to OpenAI embeddings if LLM doesn't return a valid float
                semantic_evaluator = OpenAISemanticEvaluator()
                return semantic_evaluator.compute_similarity(expected_output, actual_output)


class ConversationBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ConversationContinuation)
    
    def forward(self, conversation, user_question=None):
        # Extract the conversation up to the last user message
        user_turns = sum(1 for msg in conversation if msg["role"] == "user")
        assistant_turns = sum(1 for msg in conversation if msg["role"] == "assistant")
        
        # If this is a training example with complete conversation
        if assistant_turns >= user_turns:
            input_conv = conversation[:-1]  # Remove last assistant message
        else:
            input_conv = conversation
        if user_question is None and input_conv:
            first_user_message = next((msg for msg in input_conv if msg["role"] == "user"), None)
            if first_user_message:
                user_question = first_user_message["content"]
            else:
                user_question = ""
            
        return self.generate(conversation=input_conv, user_question=user_question)


def train_conversation_model(model_path="./model/conversation_model.json"):
    dataset = load_conversations_from_json()
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = ConversationBot()
        model.load(model_path)
        return model
    
    # Create the model
    model = ConversationBot()
    
    # Split dataset
    train_data = dataset[:5]  # First two examples for training
    test_data = dataset[5:]   # Last example for testing
    
    # Initialize a conversational evaluator with specific criteria
    conversational_evaluator = GEval(
        name="accuracy",
        criteria="""Given the 'actual output' are generated responses from an LLM chatbot, 
        'input' are user queries to the chatbot, 'expected output' is the ground
        truth, determine whether the chatbot has answered the customer's inquiry
        accurately throughout a conversation.""",
        threshold=0.5
    )
    
    # Custom metric function that uses the conversational evaluator
    def geval_metric(example, pred, trace=None):
        score = conversational_evaluator.evaluate(example, pred)
        if trace is not None:
            return score >= conversational_evaluator.threshold
        return score
    
    # Optimize using MIPROv2 with our custom metric
    optimizer = dspy.MIPROv2(metric=geval_metric)
    # optimized_model = optimizer.optimize(model, train_data=train_data, test_data=test_data)
    optimized_model = optimizer.compile(
        student=model,
        trainset=train_data,
        valset=test_data,
        minibatch_size=1,  # Set minibatch size to 1
        minibatch=True     # Enable minibatching
    )
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Saving optimized model to {model_path}...")
    optimized_model.save(model_path)
    return optimized_model


if __name__ == "__main__":
    train_conversation_model()