
import os
import glob
import yaml
import json
import shutil
import logging
from datetime import datetime
from omegaconf import DictConfig
from typing import List, Tuple
from pydantic import BaseModel
from pydantic_ai import Agent
from src.backend.utils.llm_model_factory import LLMModelFactory
from src.backend.evaluation.simulator_no_db import ChatBotSimulator
from src.creator.utils.utils import state as app_state


logger = logging.getLogger(__name__)


class Summary(BaseModel):
    title: str
    summary: str


class InputDocAgentResult(BaseModel):
    summary: List[Summary]


class PromptCreatorResult(BaseModel):
    chatbot_system_prompt: str
    simulator_system_prompt: str
    assistant_response: str
    is_complete: bool


class ReasoningCreatorResult(BaseModel):
    company_products_and_services: str
    available_information_categories: str
    data_sources: str
    current_year: int
    intent_categories_parameters: List[str]
    intent_parameters: List[str]
    examples: str


class PromptValidatorResult(BaseModel):
    new_chatbot_system_prompt: str
    new_simulator_system_prompt: str


class ConvoFeedback(BaseModel):
    customer_inquiry: str
    bot_response: str
    feedback: str


class PromptOptimizerResult(BaseModel):
    new_chatbot_prompt: str
    new_simulator_prompt: str
    convo_feedback: List[ConvoFeedback]


class ChatbotCreator():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.mode = cfg.creator.mode
        self.simulator = ChatBotSimulator(self.cfg)
        input_doc_agent_model_config = dict(self.cfg.creator.input_doc_agent)
        logger.info(f"Input doc agent model config: {input_doc_agent_model_config}")
        self.input_doc_agent_model = LLMModelFactory.create_model(input_doc_agent_model_config)
        self.input_doc_agent = Agent(
            model=self.input_doc_agent_model,
            result_type=InputDocAgentResult,
            system_prompt=self.cfg.input_doc_agent_prompts['system_prompt'],
        )
        prompt_creator_model_config = dict(self.cfg.creator.prompt_creator)
        self.prompt_creator_model = LLMModelFactory.create_model(prompt_creator_model_config)
        self.prompt_creator_agent = Agent(
            model=self.prompt_creator_model,
            result_type=PromptCreatorResult,
            system_prompt=self.cfg.prompt_creator_prompts['system_prompt'],
        )
        reasoning_creator_model_config = dict(self.cfg.creator.prompt_creator)
        self.reasoning_creator_model = LLMModelFactory.create_model(reasoning_creator_model_config)
        self.reasoning_creator_agent = Agent(
            model=self.reasoning_creator_model,
            result_type=ReasoningCreatorResult,
            system_prompt=self.cfg.reasoning_creator_prompts['system_prompt'],
        )
        prompt_validator_model_config = dict(self.cfg.creator.prompt_validator)
        self.prompt_validator_model = LLMModelFactory.create_model(prompt_validator_model_config)
        self.prompt_validator = Agent(
            model=self.prompt_validator_model,
            result_type=PromptValidatorResult,
            system_prompt=self.cfg.prompt_validator_prompts['system_prompt'],
        )
        prompt_optimizer_model_config = dict(self.cfg.creator.prompt_optimizer)
        self.prompt_optimizer_model = LLMModelFactory.create_model(prompt_optimizer_model_config)
        self.prompt_optimizer_agent = Agent(
            model=self.prompt_optimizer_model,
            result_type=PromptOptimizerResult,
            system_prompt=self.cfg.prompt_optimizer_prompts['system_prompt'],
        )

    async def summarize_input_doc(self) -> str:
        """Summarize the input document using the input_doc_agent."""
        input_doc = self.simulator.prepare_rag_context()
        formatted_input_doc = self.cfg.input_doc_agent_prompts['user_prompt'].format(input_doc=input_doc)
        logger.info(f"Formatted input doc: {formatted_input_doc}")
        result = await self.input_doc_agent.run(formatted_input_doc)
        logger.info
        parsed = InputDocAgentResult.model_validate(result.data)
        summary_dict = parsed.model_dump()
        summary_str = json.dumps(summary_dict, indent=2, ensure_ascii=False)
        return summary_str
    
    async def create_prompt(
        self, message: str, history: List[Tuple[str, str]] = None
    ) -> Tuple[str, str, str]:
        """Create a prompt using the prompt_creator_agent."""
        if history is None:
            history = []
        input_doc_summary = app_state.get("summary", "")
        if not input_doc_summary:
            raise ValueError("Document summary is required to create or optimize prompts")
        # Format conversation for the model
        conversation = ""
        for user_msg, assistant_msg in history:
            conversation += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        conversation += f"User: {message}\n"  # add current message
        
        if self.mode == "optimize":
            simulator_prompt = self.cfg.simulator_prompts['system_prompt']
            chatbot_prompt = self.cfg.query_handler_prompts['response_agent']['sys_prompt']
            reasoning_prompt = self.cfg.query_handler_prompts['reasoning_agent']['sys_prompt']

            # Store in app_state
            app_state["sim_prompt"] = simulator_prompt
            app_state["chat_prompt"] = chatbot_prompt
            app_state["reasoning_prompt"] = reasoning_prompt
            return (
                "Loaded existing prompts. Ready to proceed.",
                history,
                True,
                simulator_prompt,
                chatbot_prompt
            )

        # When mode = create
        formatted_prompt = self.cfg.prompt_creator_prompts['user_prompt'].format(
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
            simulator_prompt, chatbot_prompt = await self.validate_prompt(
                simulator_prompt=simulator_prompt,
                chatbot_prompt=chatbot_prompt
            )
            app_state["sim_prompt"] = simulator_prompt
            app_state["chat_prompt"] = chatbot_prompt
            app_state["reasoning_prompt"] = reasoning_prompt
            # update working prompt files
            self.update_prompt_files(
                simulator_prompt,
                chatbot_prompt,
                reasoning_prompt
            )
        return (
            assistant_response,
            updated_history,
            is_complete,
            simulator_prompt,
            chatbot_prompt
        )
    
    async def create_reasoning_prompt(
        self, chatbot_prompt: str, input_doc_summary: str
    ) -> str:
        """Create a reasoning prompt using the reasoning_creator_agent."""
        formatted_prompt = self.cfg.reasoning_creator_prompts['user_prompt'].format(
            chatbot_prompt=chatbot_prompt,
            input_doc_summary=input_doc_summary,
        )
        items = await self.reasoning_creator_agent.run(formatted_prompt)
        current_year = datetime.now().year
        reasoning_prompt = self.cfg.reasoning_agent_template.format(
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
        self, simulator_prompt: str, chatbot_prompt: str
    ) -> Tuple[str, str]:
        """Check the prompts using the prompt_validator_agent."""
        formatted_prompt = self.cfg.prompt_validator_prompts['user_prompt'].format(
            simulator_prompt=simulator_prompt,
            chatbot_prompt=chatbot_prompt
        )
        prompts = await self.prompt_validator.run(formatted_prompt)
        new_chatbot_system_prompt = prompts.data.new_chatbot_system_prompt
        new_simulator_system_prompt = prompts.data.new_simulator_system_prompt
        return (
            new_chatbot_system_prompt,
            new_simulator_system_prompt
        )
    
    def update_prompt_files(
        self,
        simulator_prompt: str,
        chatbot_prompt: str,
        reasoning_prompt: str = None
    ) -> None:
        """Update prompt file with the latest prompts."""
        sim_path = self.cfg.creator.simulator_prompts_filepath
        chat_path = self.cfg.creator.chatbot_prompts_filepath
        
        # Update prompts in simulator prompt file
        with open(sim_path, "r") as f:
            sim_data = yaml.safe_load(f)
        if "simulator_prompts" not in sim_data:
            sim_data["simulator_prompts"] = {}
        sim_data["simulator_prompts"]["system_prompt"] = simulator_prompt
        with open(sim_path, "w") as f:
            yaml.dump(sim_data, f)

        # Update reasoning and response prompts in query_handler prompt file
        with open(chat_path, "r") as f:
            chat_data = yaml.safe_load(f)
        chat_data["query_handler_prompts"]["response_agent"]["sys_prompt"] = chatbot_prompt
        if reasoning_prompt:
            chat_data["query_handler_prompts"]["reasoning_agent"]["sys_prompt"] = reasoning_prompt
        with open(chat_path, "w") as f:
            yaml.dump(chat_data, f)

        logger.info("📝 Updated simulator and chatbot prompt files.")

    def save_audit_log(
        self,
        session_id: str,
        iteration: int,
        new_simulator_prompt: str,
        new_chatbot_prompt: str,
        old_simulator_prompt: str,
        old_chatbot_prompt: str,
        old_reasoning_prompt: str,
        convo_feedback: List[ConvoFeedback],
        output_dir: str = "data/audit_logs"
    ) -> None:
        """Save prompts, score, feedback, and conversation."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        audit_record = {
            "session_id": session_id,
            "iteration": iteration,
            "timestamp": timestamp,
            "new_simulator_prompt": new_simulator_prompt,
            "new_chatbot_prompt": new_chatbot_prompt,
            "old_simulator_prompt": old_simulator_prompt,
            "old_chatbot_prompt": old_chatbot_prompt,
            "old_reasoning_prompt": old_reasoning_prompt,
            "convo_feedback": [
                {
                    "customer_inquiry": feedback.customer_inquiry,
                    "bot_response": feedback.bot_response,
                    "feedback": feedback.feedback
                }
                for feedback in convo_feedback
            ]
            
        }

        file_path = os.path.join(output_dir, f"{session_id}_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(audit_record, f, indent=2, ensure_ascii=False)

        logger.info(f"📦 Audit log saved to {file_path}")

    async def generate_simulations(self) -> List[dict]:
        """Generate simulation conversations with current prompts.
        This is the single point of entry for simulation generation.
        
        Returns:
            List of conversation dictionaries
        """
        # Clear the simulation directory
        simulation_dir = self.cfg.simulator.output_dir
        shutil.rmtree(simulation_dir, ignore_errors=True)
        os.makedirs(simulation_dir, exist_ok=True)
        
        # Run the simulation with current prompts
        logger.info("🔄 Running simulations with current prompts")
        await self.simulator.run_simulations(
            self.cfg.simulator.num_simulations)
        
        # Load and return the generated conversations
        conversations = []
        json_files = glob.glob(os.path.join(simulation_dir, '*.json'))
        for file_path in json_files:
            with open(file_path, 'r') as f:
                conversation = json.load(f)
                conversations.append(conversation)
        
        logger.info(f"✅ Generated {len(conversations)} conversations")
        return conversations

    async def optimize_prompts(self) -> Tuple[str, str]:
        """Optimize prompts using the prompt_optimizer_agent in a loop.
        Regenerates prompts until regenerate_prompts is False.
        Deletes previous simulation JSONs before each new simulation run.
        """
        input_doc_summary = app_state.get("summary", "")
        if not input_doc_summary:
            logger.error("No document summary found in app_state")
            raise ValueError("Document summary is required for prompt optimization")
        
        feedback_history = app_state.get("feedback_history", [])
        if not feedback_history:
            logger.error("No feedback history found in app_state")
            raise ValueError("Feedback is required for prompt optimization")
        
        # Get current prompts based on mode
        old_simulator_prompt = app_state.get("sim_prompt", self.cfg.simulator_prompts['system_prompt'])
        old_chatbot_prompt = app_state.get("chat_prompt", self.cfg.query_handler_prompts['response_agent']['sys_prompt'])
        old_reasoning_prompt = app_state.get("reasoning_prompt", self.cfg.query_handler_prompts['reasoning_agent']['sys_prompt'])
    
        # Format feedback block with text representations
        # Create ConvoFeedback objects for audit log
        feedback_list = []
        for fb in feedback_history:
            for turn in fb["conversation"]:
                feedback_obj = ConvoFeedback(
                    customer_inquiry=turn["customer_inquiry"],
                    bot_response=turn["bot_response"],
                    feedback=fb["feedback"]
                )
                feedback_list.append(feedback_obj)  # Inside the inner loop

        # Create formatted text for the prompt
        feedback_text_list = []
        for fb_obj in feedback_list:
            feedback_text_list.append(
                f"Conversation:\nUser: {fb_obj.customer_inquiry}\n"
                f"Bot: {fb_obj.bot_response}\n\n"
                f"Feedback: {fb_obj.feedback}"
            )

        feedback_block = "\n\n---\n\n".join(feedback_text_list)
        # Start Optimizing
        formatted_prompt = self.cfg.prompt_optimizer_prompts['user_prompt'].format(
            simulator_prompt=old_simulator_prompt,
            chatbot_prompt=old_chatbot_prompt,
            conversation_feedback=feedback_block
        )
        result = await self.prompt_optimizer_agent.run(formatted_prompt)
        chatbot_prompt = result.data.new_chatbot_prompt
        simulator_prompt = result.data.new_simulator_prompt

        input_doc_summary = app_state.get("summary", "")
        reasoning_prompt = await self.create_reasoning_prompt(
            chatbot_prompt=chatbot_prompt,
            input_doc_summary=input_doc_summary
        )

        # double check the new prompts to make sure they are valid
        chatbot_prompt, simulator_prompt = await self.validate_prompt(
            simulator_prompt=simulator_prompt,
            chatbot_prompt=chatbot_prompt
        )
        # Save the audit log
        session_id = f"session_iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        iteration = app_state.get("regeneration_count", 0) + 1
        self.save_audit_log(
            session_id=session_id,
            iteration=iteration,
            new_simulator_prompt=simulator_prompt,
            new_chatbot_prompt=chatbot_prompt,
            new_reasoning_prompt=reasoning_prompt,
            old_simulator_prompt=old_simulator_prompt,
            old_chatbot_prompt=old_chatbot_prompt,
            old_reasoning_prompt=old_reasoning_prompt,
            convo_feedback=feedback_list
        )
        # update working prompt files each time
        self.update_prompt_files(
            simulator_prompt,
            chatbot_prompt,
            reasoning_prompt
        )
        # Update app_state with new prompts
        app_state["sim_prompt"] = simulator_prompt
        app_state["chat_prompt"] = chatbot_prompt
        app_state["reasoning_prompt"] = reasoning_prompt
    
        logger.info("✅ Prompts optimized")
        logger.info(f"📝 Feedback count: {len(feedback_text_list)}")

        return chatbot_prompt, simulator_prompt

