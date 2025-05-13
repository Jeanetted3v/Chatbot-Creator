import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from src.backend.utils.llm_model_factory import LLMModelFactory

logger = logging.getLogger(__name__)


class PromptCreatorResult(BaseModel):
    chatbot_system_prompt: Optional[str]
    simulator_system_prompt: Optional[str]
    assistant_response: str
    is_complete: bool


class ReasoningCreatorResult(BaseModel):
    company_products_and_services: str
    available_information_categories: str
    data_sources: str
    intent_categories: str
    intent_parameters: str
    examples: str


class PromptValidatorResult(BaseModel):
    new_chatbot_system_prompt: str
    new_simulator_system_prompt: str


class PromptCreationService:
    """Service for creating and validating prompts through interactive conversation"""
    
    def __init__(self, config_service):
        self.config_service = config_service
        self._init_agents()
    
    def _init_agents(self):
        """Initialize all creation-related agents"""
        cfg = self.config_service.get_config()
        
        # Prompt creator agent
        creator_model_config = dict(cfg.creator.prompt_creator)
        self.creator_model = LLMModelFactory.create_model(creator_model_config)
        self.prompt_creator_agent = Agent(
            model=self.creator_model,
            result_type=PromptCreatorResult,
            system_prompt=cfg.prompt_creator_prompts['system_prompt'],
        )
        
        # Reasoning creator agent
        reasoning_model_config = dict(cfg.creator.reasoning_creator)
        self.reasoning_model = LLMModelFactory.create_model(reasoning_model_config)
        self.reasoning_creator_agent = Agent(
            model=self.reasoning_model,
            result_type=ReasoningCreatorResult,
            system_prompt=cfg.reasoning_creator_prompts['system_prompt'],
        )
        
        # Prompt validator agent
        validator_model_config = dict(cfg.creator.prompt_validator)
        self.validator_model = LLMModelFactory.create_model(validator_model_config)
        self.prompt_validator = Agent(
            model=self.validator_model,
            result_type=PromptValidatorResult,
            system_prompt=cfg.prompt_validator_prompts['system_prompt'],
        )
    
    async def create_prompt(
        self,
        message: str,
        input_doc_summary: str,
        history: List[Tuple[str, str]] = None
    ) -> Tuple[str, str, bool, str, str]:
        """
        Create prompts through interactive conversation with the user.
        """
        cfg = self.config_service.get_config()
        history = history if history else []
        
        # Format conversation for the model
        conversation = ""
        for user_msg, assistant_msg in history:
            conversation += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        conversation += f"User: {message}\n"
        
        # When in optimize mode, just return existing prompts
        if cfg.creator.mode == "optimize":
            simulator_prompt = cfg.simulator_prompts['system_prompt']
            chatbot_prompt = cfg.query_handler_prompts['response_agent']['sys_prompt']
            reasoning_prompt = cfg.query_handler_prompts['reasoning_agent']['sys_prompt']
            
            return (
                "Loaded existing prompts. Ready to proceed.",
                history,
                True,
                simulator_prompt,
                chatbot_prompt
            )

        # When in create mode
        formatted_prompt = cfg.prompt_creator_prompts['user_prompt'].format(
            input_doc_summary=input_doc_summary,
            conversation=conversation,
        )
        
        result = await self.prompt_creator_agent.run(formatted_prompt)
        simulator_prompt = result.data.simulator_system_prompt
        chatbot_prompt = result.data.chatbot_system_prompt
        assistant_response = result.data.assistant_response
        is_complete = result.data.is_complete
        updated_history = history + [(message, assistant_response)]

        if is_complete:
            logger.info("Prompt creation completed successfully.")
            reasoning_prompt = await self.create_reasoning_prompt(
                chatbot_prompt=chatbot_prompt,
                input_doc_summary=input_doc_summary
            )
            
            chatbot_prompt, simulator_prompt = await self.validate_prompt(
                simulator_prompt=simulator_prompt,
                chatbot_prompt=chatbot_prompt
            )
            
            # Update prompts in configuration files
            self.config_service.update_prompts(
                simulator_prompt=simulator_prompt,
                chatbot_prompt=chatbot_prompt,
                reasoning_prompt=reasoning_prompt
            )
            
        return (
            assistant_response,
            updated_history,
            is_complete,
            simulator_prompt,
            chatbot_prompt
        )
    
    async def create_reasoning_prompt(
        self, 
        chatbot_prompt: str, 
        input_doc_summary: str
    ) -> str:
        """Create reasoning prompt from chatbot prompt and document summary"""
        cfg = self.config_service.get_config()
        
        formatted_prompt = cfg.reasoning_creator_prompts['user_prompt'].format(
            chatbot_prompt=chatbot_prompt,
            input_doc_summary=input_doc_summary,
        )
        
        items = await self.reasoning_creator_agent.run(formatted_prompt)
        logger.info(f"Items from reasoning agent creator: {items.data}")
        current_year = datetime.now().year

        reasoning_template = cfg.reasoning_agent_template
        logger.info(f"Reasoning template: {reasoning_template}")
        logger.info(f"Items data: {items.data}")
        reasoning_prompt = reasoning_template.format(
            company_products_and_services=items.data.company_products_and_services,
            available_information_categories=items.data.available_information_categories,
            data_sources=items.data.data_sources,
            current_year=current_year,
            intent_categories=items.data.intent_categories,
            intent_parameters=items.data.intent_parameters,
            examples=items.data.examples,
        )
        
        return reasoning_prompt

    async def validate_prompt(
        self,
        simulator_prompt: str,
        chatbot_prompt: str
    ) -> Tuple[str, str]:
        """Validate and improve prompts"""
        cfg = self.config_service.get_config()
        
        formatted_prompt = cfg.prompt_validator_prompts['user_prompt'].format(
            simulator_prompt=simulator_prompt,
            chatbot_prompt=chatbot_prompt
        )
        
        prompts = await self.prompt_validator.run(formatted_prompt)
        
        return (
            prompts.data.new_chatbot_system_prompt,
            prompts.data.new_simulator_system_prompt
        )