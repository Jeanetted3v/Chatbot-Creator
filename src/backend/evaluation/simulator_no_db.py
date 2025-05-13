"""To simulate a conversation with a simulated user between the
chatbot and a LLM simulated user. Not saving to MongoDB.
To run:
python -m src.backend.prompt_optimizer.simulator
"""
import logging
import logfire
import asyncio
import uuid
import hydra
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from src.backend.utils.logging import setup_logging
from src.backend.utils.llm_model_factory import LLMModelFactory
from src.backend.chat.hybrid_retriever import HybridRetriever
from src.backend.dataloaders.local_doc_loader import load_local_doc, LoadedUnstructuredDocument, LoadedStructuredDocument


logger = logging.getLogger(__name__)
logger.info('Setting up logging configuration.')
setup_logging()
logfire.configure(send_to_logfire='if-token-present')


class ConversationHistory:
    """Conversation history tracking"""
    def __init__(self):
        self.messages = []
        self.exchanges = []

    def add_message(self, role: str, content: str):
        self.messages.append({'role': role, 'content': content})

    def add_exchange(self, inquiry: str, response: str, retrieval_context: str=''):
        self.exchanges.append({'customer_inquiry': inquiry, 'bot_response': response, 'retrieval_context': retrieval_context})

    async def format_history_for_prompt(self) -> str:
        formatted = []
        for msg in self.messages:
            prefix = 'User: ' if msg['role'] == 'user' else 'Assistant: '
            formatted.append(f"{prefix}{msg['content']}")
        return '\n'.join(formatted)

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def get_exchanges(self) -> List[Dict[str, str]]:
        return self.exchanges


class ReasoningResult(BaseModel):
    """Result model for reasoning agent"""
    expanded_query: List[str]
    need_search: bool


class QueryHandlerResponse(BaseModel):
    """Response model for query handler"""
    response: str
    intent: str
    english_level: Optional[str] = None
    course_interest: Optional[str] = None
    lexile_level: Optional[str] = None


class LLMGroundTruth(BaseModel):
    """Model for a single LLM ground truth item"""
    customer_inquiry: str
    llm_gt: str


class AllLLMGroundTruth(BaseModel):
    """Model for all LLM ground truths"""
    all_llmgt: List[LLMGroundTruth] = Field(default_factory=list)


class ChatBotSimulator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.hybrid_retriever = HybridRetriever(self.cfg)
        self.documents = load_local_doc(cfg)
        self.rag_context = self.prepare_rag_context()
        chatbot_model_config = dict(self.cfg.simulator.chatbot_llm)
        self.chatbot_model = LLMModelFactory.create_model(chatbot_model_config)
        logger.info(f'Chatbot LLM model instance created: {self.chatbot_model}')
        user_model_config = dict(self.cfg.simulator.user_llm)
        self.user_model = LLMModelFactory.create_model(user_model_config)
        logger.info(f'User simulation LLM model instance created: {self.user_model}')
        gt_model_config = dict(self.cfg.simulator.gt_llm)
        self.gt_model = LLMModelFactory.create_model(gt_model_config)
        logger.info(f'Ground truth LLM model instance created: {self.gt_model}')
        self.reasoning_agent = Agent(
            model=self.chatbot_model,
            result_type=ReasoningResult,
            system_prompt=self.cfg.query_handler_prompts.reasoning_agent['sys_prompt']
        )
        self.response_agent = Agent(
            model=self.chatbot_model,
            result_type=QueryHandlerResponse,
            system_prompt=self.cfg.query_handler_prompts.response_agent['sys_prompt']
        )
        self.simulator_agent = Agent(
            model=self.user_model,
            result_type=str,
            system_prompt=self.cfg.simulator_prompts.system_prompt
        )
        self.gt_agent = Agent(
            model=self.gt_model,
            result_type=AllLLMGroundTruth,
            system_prompt=self.cfg.llm_gt_prompts.system_prompt
        )

    def prepare_rag_context(self) -> str:
        """Prepare RAG context from loaded documents"""
        rag_context = ''
        for doc in self.documents:
            if isinstance(doc, LoadedUnstructuredDocument):
                rag_context += f'\n\n{doc.content}'
            else:  # inserted
                if isinstance(doc, LoadedStructuredDocument):
                    df_str = doc.content.to_string(index=False)
                    rag_context += f'\n\n{df_str}'
        return rag_context

    def print_conversation(self, role: str, content: str) -> None:
        """Print conversation with appropriate role labels"""
        if role == 'user':
            print(f'\nSimulated User: {content}')
        else:  # inserted
            print(f'\n{role.capitalize()}: {content}')

    async def get_simulated_user_query(self, last_bot_response: str, history: ConversationHistory) -> str:
        """Get next query from LLM simulator"""  # inserted
        history_str = await history.format_history_for_prompt()
        try:
            next_user_query = await self.simulator_agent.run(
                self.cfg.simulator_prompts.user_prompt.format(
                    last_bot_response=last_bot_response,
                    msg_history=history_str,
                    exchange_limit=self.cfg.simulator.max_exchange_limit
                )
            )
            return next_user_query.data
        except Exception as e:
            logger.error(f'Error generating simulated query: {e}')
            return 'Could you explain that again? I didn\'t understand.'

    async def generate_chatbot_response(
        self, query: str, history: ConversationHistory
    ) -> tuple:
        """Generate a chatbot response with retrieval context"""
        try:
            msg_history = await history.format_history_for_prompt()
            reasoning_result = await self.reasoning_agent.run(
                self.cfg.query_handler_prompts.reasoning_agent['user_prompt'].format(
                    query=query,
                    message_history=msg_history,
                    competitors=self.cfg.guardrails.competitors
                )
            )
            need_search = reasoning_result.data.need_search
            all_search_results = []
            retrieval_context = ''
            if need_search:
                for eq in reasoning_result.data.expanded_query:
                    search_results = await self.hybrid_retriever.search(eq)
                    all_search_results.append(search_results)
                if all_search_results and len(all_search_results) > 0 and (len(all_search_results[0]) > 0):
                    retrieval_context = all_search_results[0][0]
            result = await self.response_agent.run(
                self.cfg.query_handler_prompts.response_agent['user_prompt'].format(
                    query=query,
                    message_history=msg_history,
                    search_results=all_search_results,
                    competitors=self.cfg.guardrails.competitors
                )
            )
            response = result.data.response
            return (response, retrieval_context)
        except Exception as e:
            logger.error(f'Error in query handling: {e}')
            return ('I\'m sorry, I encountered an error processing your request.', '')

    async def generate_ground_truth(self, inquiries: List[str]) -> List[LLMGroundTruth]:
        """Generate ground truth answers for given inquiries"""  # inserted
        try:
            formatted_inquiries = '\n'.join([f'Inquiry {i + 1}: {inq}' for i, inq in enumerate(inquiries)])
            result = await self.gt_agent.run(
                self.cfg.llm_gt_prompts.user_prompt.format(
                    customer_inquiry=formatted_inquiries,
                    context=self.rag_context
                )
            )
            return result.data.all_llmgt
        except Exception as e:
            logger.error(f'Error generating ground truth: {e}')
            return []

    async def process_query(self, query: str, history: ConversationHistory) -> str:
        """Process user query"""  # inserted
        self.print_conversation('user', query)
        history.add_message('user', query)
        response, retrieval_context = await self.generate_chatbot_response(query, history)
        history.add_message('assistant', response)
        history.add_exchange(query, response, retrieval_context)
        self.print_conversation('assistant', response)
        return response

    async def save_conversation_to_json(
        self,
        session_id: str,
        exchanges: List[Dict],
        ground_truths: List[LLMGroundTruth]
    ) -> None:
        """Save conversation to a JSON file in the specified format"""
        output_dir = self.cfg.simulator.output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{output_dir}/{session_id}_{timestamp}.json'
        conversation_data = []
        for i, exchange in enumerate(exchanges):
            retrieval_ctx = exchange.get('retrieval_context', '')
            if hasattr(retrieval_ctx, 'model_dump'):
                retrieval_ctx = retrieval_ctx.model_dump()
                context = retrieval_ctx.get('content', '')
            entry = {'session_id': session_id, 'customer_inquiry': exchange['customer_inquiry'], 'bot_response': exchange['bot_response'], 'retrieval_context': context, 'context': '', 'expected_output': ''}
            if i < len(ground_truths):
                entry['expected_output'] = ground_truths[i].llm_gt
            conversation_data.append(entry)
        with open(filename, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        logger.info(f'Saved conversation to {filename}')

    async def run_single_simulation(self, simulation_index: int) -> str:
        """Run a single simulation"""  # inserted
        session_id = f'session_{uuid.uuid4()}'
        history = ConversationHistory()
        try:
            simulation_mode = True
            last_bot_response = ''
            first_query = self.cfg.simulator.first_query
            last_bot_response = await self.process_query(first_query, history)
        
            exchange_count = 1
            while simulation_mode and exchange_count < self.cfg.simulator.max_exchange_limit:
                query = await self.get_simulated_user_query(
                    last_bot_response, history)
                if any((word in query.lower() for word in ['bye', 'goodbye'])):
                    history.add_message('user', query)
                    last_bot_response, retrieval_context = await self.generate_chatbot_response(query, history)
                    history.add_message('assistant', last_bot_response)
                    history.add_exchange(query, last_bot_response, retrieval_context)
                    self.print_conversation('assistant', last_bot_response)
                    simulation_mode = False
                    break
                else:
                    last_bot_response = await self.process_query(query, history)
                    exchange_count += 1
            inquiries = [ex['customer_inquiry'] for ex in history.get_exchanges()]
            ground_truths = await self.generate_ground_truth(inquiries)
            await self.save_conversation_to_json(session_id, history.get_exchanges(), ground_truths)
            return session_id
        except Exception as e:
            logger.error(f'Error in simulation {simulation_index}: {e}')
            return None

    async def run_simulations(self, num_simulations: int) -> None:
        """Run multiple simulations in parallel"""  # inserted
        try:
            tasks = []
            for i in range(num_simulations):
                task = asyncio.create_task(self.run_single_simulation(i + 1))
                tasks.append(task)
            completed_sessions = await asyncio.gather(*tasks)
            logger.info(f'Completed {num_simulations} simulations. Session IDs: {completed_sessions}')
        except Exception as e:
            logger.error(f'Error running simulations: {e}')
            return
        
    def extract_customer_bot(
        self, json_file_paths: List[str]
    ) -> List[Dict[str, str]]:
        """Extract customer inquiries and bot responses from a json file
        To use in prompt creation process.
        """
        all_turns = []
        for path in json_file_paths:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                for entry in data:
                    turn = {
                        'Customer': entry.get('customer_inquiry', '').strip(),
                        'Bot': entry.get('bot_response', '').strip(),
                        'session_id': entry.get('session_id', '')
                    }
                    all_turns.append(turn)
            except Exception as e:
                print(f"⚠️ Error reading {path}: {e}")
        return all_turns



@hydra.main(
    version_base=None,
    config_path='../../../config',
    config_name='config'
)
def main(cfg) -> None:
    async def async_main():
        simulator = ChatBotSimulator(cfg)
        await simulator.run_simulations(cfg.simulator.num_simulations)
    asyncio.run(async_main())


if __name__ == '__main__':
    main()