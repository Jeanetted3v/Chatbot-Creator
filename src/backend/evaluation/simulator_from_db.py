"""To run:
python -m src.backend.evaluation.simulator
"""
import logging
import logfire
import asyncio
import uuid
import hydra

from omegaconf import DictConfig
from pydantic_ai import Agent
from src.backend.utils.logging import setup_logging
from src.backend.chat.service_container import ServiceContainer
from src.backend.utils.llm_model_factory import LLMModelFactory


logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()
logfire.configure(send_to_logfire='if-token-present')


class Simulator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.services = ServiceContainer(cfg)
        self.session_id = f"session_{uuid.uuid4()}"
        self.customer_id = f"customer_{uuid.uuid4()}"
        self.agent_mode = "bot"
        model_config = dict(self.cfg.simulator.llm)
        model = LLMModelFactory.create_model(model_config)
        logger.info(f"LLM model instance created: {model}")
        self.prompts = cfg.simulator_prompts
        self.simulator_agent = Agent(
            model=model,
            result_type=str,
            system_prompt=self.prompts['sys_prompt'],
        )
        self.simulation_mode = cfg.simulator.enabled

    def print_conversation(self, role: str, content: str) -> None:
        """Print conversation with appropriate role labels"""
        if role == 'user':
            print(f"\nSimulated User: {content}")
        else:
            print(f"\n{role.capitalize()}: {content}")
        
    async def get_simulated_user_query(self, last_bot_response: str) -> str:
        """Get next query from LLM simulator"""
        chat_history = await self.services.get_chat_history(
            self.session_id,
            self.customer_id
        )
        # Format history for the simulator
        history_str = await chat_history.format_history_for_prompt()
        logger.info(f"Chat history for simulation: {history_str}")

        try:
            next_user_query = await self.simulator_agent.run(
                self.prompts['user_prompt'].format(
                    last_bot_response=last_bot_response,
                    msg_history=history_str,
                    exchange_limit=self.cfg.simulator.max_exchange_limit,
                )
            )
            next_user_query = next_user_query.data
            return next_user_query
        except Exception as e:
            logger.error(f"Error generating simulated query: {e}")
            return "Could you explain that again? I didn't understand."
    
    async def process_query(self, query: str) -> str:
        """Process user query"""
        self.print_conversation("user", query)
        response = await self.services.query_handler.handle_query(
            query,
            self.session_id,
            self.customer_id
        )
        self.print_conversation("assistant", response)
        return response

    async def run_simulations(self, num_simulations: int = 1) -> None:
        """Run multiple simulations"""
        await self.services.initialize()
        try:
            for i in range(num_simulations):
                # Create new session for each simulation
                self.session_id = f"session_{uuid.uuid4()}"
                self.customer_id = f"customer_{uuid.uuid4()}"
                await self.services.get_or_create_session(
                    self.session_id, self.customer_id
                )
                self.simulation_mode = True
                last_bot_response = ""
                first_query = "Hi, I'm a parent, would like to make an inquiry"
                last_bot_response = await self.process_query(first_query)
                while self.simulation_mode:
                    query = await self.get_simulated_user_query(
                        last_bot_response)
                    
                    if any(word in query.lower() for word in ["bye"]):
                        print("\nSimulation ended by LLM.")
                        self.simulation_mode = False
                        break
                    
                    last_bot_response = await self.process_query(query)
        finally:
            await self.services.cleanup()


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="config")
def main(cfg) -> None:
    
    async def async_main():
        simulator = Simulator(cfg)
        await simulator.run_simulations(cfg.simulator.num_simulations)
    
    asyncio.run(async_main())


if __name__ == "__main__":
    main()