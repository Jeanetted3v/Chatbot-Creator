import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from src.backend.utils.llm_model_factory import LLMModelFactory

logger = logging.getLogger(__name__)


class ConvoFeedback(BaseModel):
    customer_inquiry: str
    bot_response: str
    feedback: str


class PromptOptimizerResult(BaseModel):
    new_chatbot_prompt: str
    new_simulator_prompt: str
    convo_feedback: List[ConvoFeedback]


class PromptOptimizationService:
    """Service for optimizing prompts based on feedback from simulated conversations"""
    
    def __init__(self, config_service, prompt_creation_service):
        self.config_service = config_service
        self.prompt_creation_service = prompt_creation_service
        self._init_agents()
    
    def _init_agents(self):
        """Initialize optimization agent"""
        cfg = self.config_service.get_config()
        
        # Prompt optimizer agent
        optimizer_model_config = dict(cfg.creator.prompt_optimizer)
        self.optimizer_model = LLMModelFactory.create_model(optimizer_model_config)
        self.prompt_optimizer_agent = Agent(
            model=self.optimizer_model,
            result_type=PromptOptimizerResult,
            system_prompt=cfg.prompt_optimizer_prompts['system_prompt'],
        )
    
    async def optimize_prompts(
        self,
        input_doc_summary: str,
        feedback_history: List[Dict],
        current_prompts: Dict[str, str] = None
    ) -> Tuple[str, str, str, List[ConvoFeedback]]:
        """
        Optimize prompts based on feedback from simulated conversations.
        
        Args:
            input_doc_summary: Summary of the input document
            feedback_history: List of feedback items from simulated conversations
            current_prompts: Dict containing current prompts (optional)
            
        Returns:
            Tuple of (chatbot_prompt, simulator_prompt, reasoning_prompt, feedback_list)
        """
        cfg = self.config_service.get_config()
        
        # Get current prompts
        if current_prompts is None:
            current_prompts = {
                "simulator": cfg.simulator_prompts['system_prompt'],
                "chatbot": cfg.query_handler_prompts['response_agent']['sys_prompt'],
                "reasoning": cfg.query_handler_prompts['reasoning_agent']['sys_prompt']
            }
        
        # Format feedback and create ConvoFeedback objects for audit
        feedback_list = []
        feedback_text_list = []
        
        for fb in feedback_history:
            for turn in fb["conversation"]:
                feedback_obj = ConvoFeedback(
                    customer_inquiry=turn["customer_inquiry"],
                    bot_response=turn["bot_response"],
                    feedback=fb["feedback"]
                )
                feedback_list.append(feedback_obj)
                
                feedback_text_list.append(
                    f"Conversation:\nUser: {feedback_obj.customer_inquiry}\n"
                    f"Bot: {feedback_obj.bot_response}\n\n"
                    f"Feedback: {feedback_obj.feedback}"
                )

        # Join feedback block with separators
        feedback_block = "\n\n---\n\n".join(feedback_text_list)
        
        # Format and run the optimization prompt
        formatted_prompt = cfg.prompt_optimizer_prompts['user_prompt'].format(
            simulator_prompt=current_prompts["simulator"],
            chatbot_prompt=current_prompts["chatbot"],
            conversation_feedback=feedback_block
        )
        
        result = await self.prompt_optimizer_agent.run(formatted_prompt)
        
        # Get optimized prompts
        chatbot_prompt = result.data.new_chatbot_prompt
        simulator_prompt = result.data.new_simulator_prompt
        
        # Create a new reasoning prompt based on the optimized chatbot prompt
        reasoning_prompt = await self.prompt_creation_service.create_reasoning_prompt(
            chatbot_prompt=chatbot_prompt,
            input_doc_summary=input_doc_summary
        )
        
        # Validate the optimized prompts
        chatbot_prompt, simulator_prompt = await self.prompt_creation_service.validate_prompt(
            simulator_prompt=simulator_prompt,
            chatbot_prompt=chatbot_prompt
        )
        
        # Save audit log
        self.save_audit_log(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            iteration=1,  # Get from app_state or pass as parameter
            prompts={
                "new": {
                    "simulator": simulator_prompt,
                    "chatbot": chatbot_prompt,
                    "reasoning": reasoning_prompt
                },
                "old": current_prompts
            },
            feedback_list=feedback_list
        )
        
        # Update prompts in configuration
        self.config_service.update_prompts(
            simulator_prompt=simulator_prompt,
            chatbot_prompt=chatbot_prompt,
            reasoning_prompt=reasoning_prompt
        )
        
        logger.info("✅ Prompts optimized")
        logger.info(f"📝 Feedback count: {len(feedback_text_list)}")
        
        return chatbot_prompt, simulator_prompt, reasoning_prompt, feedback_list
    
    def save_audit_log(
        self,
        session_id: str,
        iteration: int,
        prompts: Dict,
        feedback_list: List[ConvoFeedback],
        output_dir: str = "data/audit_logs"
    ) -> None:
        """Save prompts, feedback, and other data to audit log"""
        cfg = self.config_service.get_config()
        output_dir = cfg.get("audit_log_dir", output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        audit_record = {
            "session_id": session_id,
            "iteration": iteration,
            "timestamp": timestamp,
            "new_simulator_prompt": prompts["new"]["simulator"],
            "new_chatbot_prompt": prompts["new"]["chatbot"],
            "new_reasoning_prompt": prompts["new"]["reasoning"],
            "old_simulator_prompt": prompts["old"]["simulator"],
            "old_chatbot_prompt": prompts["old"]["chatbot"],
            "old_reasoning_prompt": prompts["old"]["reasoning"],
            "convo_feedback": [
                {
                    "customer_inquiry": feedback.customer_inquiry,
                    "bot_response": feedback.bot_response,
                    "feedback": feedback.feedback
                }
                for feedback in feedback_list
            ]
        }
        
        file_path = os.path.join(output_dir, f"{session_id}_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(audit_record, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📦 Audit log saved to {file_path}")