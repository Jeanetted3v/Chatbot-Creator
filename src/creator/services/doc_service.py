import json
import logging
from typing import List, Dict
from pydantic import BaseModel
from pydantic_ai import Agent
from src.backend.utils.llm_model_factory import LLMModelFactory

logger = logging.getLogger(__name__)


class Summary(BaseModel):
    title: str
    summary: str


class InputDocAgentResult(BaseModel):
    summary: List[Summary]


class DocumentService:
    """Service for document summarization and processing"""
    
    def __init__(self, config_service, simulation_service):
        self.config_service = config_service
        self.simulation_service = simulation_service
        self._init_agent()
    
    def _init_agent(self):
        """Initialize the document summarization agent"""
        cfg = self.config_service.get_config()
        input_doc_agent_model_config = dict(cfg.creator.input_doc_agent)
        logger.info(f"Input doc agent model config: {input_doc_agent_model_config}")
        
        self.input_doc_agent_model = LLMModelFactory.create_model(input_doc_agent_model_config)
        self.input_doc_agent = Agent(
            model=self.input_doc_agent_model,
            result_type=InputDocAgentResult,
            system_prompt=cfg.input_doc_agent_prompts['system_prompt'],
        )
    
    async def summarize_document(self) -> str:
        """
        Summarize the input document using the input_doc_agent.
        
        Returns:
            String representation of the document summary
        """
        cfg = self.config_service.get_config()
        input_doc = self.simulation_service.prepare_rag_context()
        formatted_input_doc = cfg.input_doc_agent_prompts['user_prompt'].format(
            input_doc=input_doc
        )
        logger.info("Summarizing document...")
        result = await self.input_doc_agent.run(formatted_input_doc)
        parsed = InputDocAgentResult.model_validate(result.data)
        summary_dict = parsed.model_dump()
        summary_str = json.dumps(summary_dict, indent=2, ensure_ascii=False)
        return summary_str