import os
import glob
import json
import shutil
import logging
from typing import List, Dict, Optional
from src.backend.evaluation.simulator_no_db import ChatBotSimulator

logger = logging.getLogger(__name__)


class SimulationService:
    """Service for generating simulated conversations with the chatbot"""
    
    def __init__(self, config_service):
        self.config_service = config_service
        self.simulator = None
        self._init_simulator()
    
    def _init_simulator(self):
        """Initialize the simulator with current configuration"""
        cfg = self.config_service.get_config()
        self.simulator = ChatBotSimulator(cfg)
    
    async def generate_simulations(
        self, 
        prompts: Optional[Dict[str, str]] = None,
        num_simulations: int = None
    ) -> List[dict]:
        """
        Generate simulation conversations with current prompts.
        
        Args:
            prompts: Optional dict with simulator, chatbot, and reasoning prompts
            num_simulations: Number of simulations to run (default from config)
            
        Returns:
            List of conversation dictionaries
        """
        cfg = self.config_service.get_config()
        
        # If prompts are provided, update the config
        if prompts:
            self.config_service.update_prompts(
                simulator_prompt=prompts.get("simulator"),
                chatbot_prompt=prompts.get("chatbot"),
                reasoning_prompt=prompts.get("reasoning")
            )
            # Reinitialize simulator with updated config
            self._init_simulator()
        
        # Get simulation directory from config
        simulation_dir = cfg.simulator.output_dir
        
        # Use specified num_simulations or get from config
        sim_count = num_simulations or cfg.simulator.num_simulations
        
        # Clear the simulation directory
        shutil.rmtree(simulation_dir, ignore_errors=True)
        os.makedirs(simulation_dir, exist_ok=True)
        
        # Run the simulation with current prompts
        logger.info(f"🔄 Running {sim_count} simulations with current prompts")
        await self.simulator.run_simulations(sim_count)
        
        # Load and return the generated conversations
        conversations = []
        json_files = glob.glob(os.path.join(simulation_dir, '*.json'))
        for file_path in json_files:
            with open(file_path, 'r') as f:
                conversation = json.load(f)
                conversations.append(conversation)
        
        logger.info(f"✅ Generated {len(conversations)} conversations")
        return conversations
    
    def prepare_rag_context(self) -> str:
        """
        Prepare the RAG context for document summarization.
        This is used by the document service when summarizing input documents.
        
        Returns:
            String containing the RAG context
        """
        if self.simulator is None:
            self._init_simulator()
        
        return self.simulator.prepare_rag_context()