# src/creator/services/config_service.py
"""Configuration service to manage config changes and propagation"""
import yaml
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigService:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._save_paths = {
            "simulator_prompts": cfg.creator.simulator_prompts_filepath,
            "chatbot_prompts": cfg.creator.chatbot_prompts_filepath
        }
    
    def get_config(self) -> DictConfig:
        """Get the current configuration"""
        return self.cfg
    
    def update_model(
        self, reasoning_model: str, response_model: str
    ) -> None:
        """Update model configuration across all services"""
        self.cfg.reasoning.model_name = reasoning_model
        self.cfg.response.model_name = response_model
        logger.info(f"Updated model to Reasoning: {reasoning_model} and "
                    f"Response: {response_model}")
    
    def get_current_prompts(self):
        """Get the current prompts from config"""
        return {
            "simulator": self.cfg.simulator_prompts['system_prompt'],
            "chatbot": self.cfg.query_handler_prompts['response_agent']['sys_prompt'],
            "reasoning": self.cfg.query_handler_prompts['reasoning_agent']['sys_prompt']
        }

    def update_prompts(
        self,
        simulator_prompt: str,
        chatbot_prompt: str,
        reasoning_prompt: Optional[str] = None
    ) -> None:
        """Update prompts in config and save to files"""
        # Update in-memory config
        self.cfg.simulator_prompts.system_prompt = simulator_prompt
        self.cfg.query_handler_prompts.response_agent.sys_prompt = chatbot_prompt
        if reasoning_prompt:
            self.cfg.query_handler_prompts.reasoning_agent.sys_prompt = reasoning_prompt
        
        # Save to files
        self._save_prompt_files(simulator_prompt, chatbot_prompt, reasoning_prompt)
        logger.info("Updated prompts in config and saved to files")
    
    def _save_prompt_files(
        self,
        simulator_prompt: str,
        chatbot_prompt: str,
        reasoning_prompt: Optional[str] = None
    ) -> None:
        """Save prompts to YAML files"""
        # Save simulator prompt
        with open(self._save_paths["simulator_prompts"], "r") as f:
            sim_data = yaml.safe_load(f)
        
        if "simulator_prompts" not in sim_data:
            sim_data["simulator_prompts"] = {}
        sim_data["simulator_prompts"]["system_prompt"] = simulator_prompt
        
        with open(self._save_paths["simulator_prompts"], "w") as f:
            yaml.dump(sim_data, f)
        
        # Save chatbot and reasoning prompts
        with open(self._save_paths["chatbot_prompts"], "r") as f:
            chat_data = yaml.safe_load(f)
        
        chat_data["query_handler_prompts"]["response_agent"]["sys_prompt"] = chatbot_prompt
        if reasoning_prompt:
            chat_data["query_handler_prompts"]["reasoning_agent"]["sys_prompt"] = reasoning_prompt
        
        with open(self._save_paths["chatbot_prompts"], "w") as f:
            yaml.dump(chat_data, f)