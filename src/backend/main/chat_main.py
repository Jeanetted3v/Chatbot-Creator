"""To interact with this chatbot via terminal, run:
python -m src.backend.main.chat_main
"""
import logging
import logfire
import hydra
import json
import traceback
import asyncio
import uuid
from omegaconf import DictConfig

from src.backend.utils.logging import setup_logging
from src.backend.chat.service_container import ServiceContainer

logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()
logfire.configure(send_to_logfire='if-token-present')


class CLITester:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.services = ServiceContainer(cfg)
        self.session_id = f"session_{uuid.uuid4()}"
        self.customer_id = f"customer_{uuid.uuid4()}"
        self.agent_mode = "bot"  # Track current mode
    
    async def initialize(self):
        """Async initialization method"""
        await self.services.initialize()
        await self.services.get_or_create_session(
            self.session_id, self.customer_id
        )
        logger.info(f"Session created with ID: {self.session_id}, "
                    f"Customer ID: {self.customer_id}")

    def print_conversation(self, role: str, content: str) -> None:
        """Print conversation with appropriate role labels"""
        if role == 'system':
            print(f"\n[System]: {content}")
        elif role == 'human_agent':
            print(f"\nHuman Agent: {content}")
        else:
            print(f"\n{role.capitalize()}: {content}")

    async def handle_human_agent_response(self, query: str) -> str:
        """Mock human agent responses for testing"""
        if "bye" in query.lower():
            self.agent_mode = "bot"
            chat_history = await self.services.get_chat_history(
                self.session_id,
                self.customer_id
            )
            await self.services.human_handler.transfer_to_bot(
                self.session_id,
                chat_history
            )
            return "Transferring back to AI assistant. Have a great day!"
        return f"[Human Agent Response] I understand your query about: {query}"

    async def process_query(self, query: str) -> None:
        """Process user query based on current mode"""
        self.print_conversation("user", query)
        chat_history = await self.services.get_chat_history(
            self.session_id,
            self.customer_id
        )
        
        if self.agent_mode == "human":
            response = await self.handle_human_agent_response(query)
            await chat_history.add_turn('human_agent', response)
            self.print_conversation("human_agent", response)
            return

        response = await self.services.query_handler.handle_query(
            query,
            self.session_id,
            self.customer_id
        )
        
        # Check if response indicates transfer to human
        if "Transferring" in response and "human agent" in response:
            self.agent_mode = "human"
            
        self.print_conversation(
            "human_agent" if self.agent_mode == "human" else "assistant",
            response
        )

    async def run(self) -> None:
        """Run the CLI tester"""
        try:
            await self.initialize()
            print("\nWelcome to the Edu Chatbot Tester!")
            print("Commands:")
            print("- 'quit' or 'exit': End session")
            print("- 'stats': Show current session stats")

            while True:
                try:
                    query = input("\nUser: ").strip()

                    if query.lower() in ['quit', 'exit']:
                        print("\nGoodbye!")
                        break
                    elif query.lower() == 'stats':
                        stats = self.services.human_handler.get_session_stats(
                            self.session_id
                        )
                        print("\nSession Stats:", json.dumps(stats, indent=2))
                        continue
                    elif not query:
                        continue

                    await self.process_query(query)

                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    logging.error(f"Error processing query: {e}")
                    print(f"\nError: {e}")
                    traceback.print_exc()
        finally:
            await self.services.cleanup()


@hydra.main(
    version_base=None,
    config_path="../../../config",
    config_name="config")
def main(cfg) -> None:
    
    async def async_main():
        tester = CLITester(cfg)
        await tester.run()
    
    asyncio.run(async_main())


if __name__ == "__main__":
    main()