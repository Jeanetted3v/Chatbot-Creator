from typing import List, Dict
import os
import json
import csv
import logging
from datetime import datetime
from src.backend.utils.settings import SETTINGS
from src.backend.chat.service_container import ServiceContainer

# Import RAGAS components
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall
)
from ragas.llms import OpenAI as RagasOpenAI
from datasets import Dataset

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """RAGAS evaluator for your chatbot system."""
    
    def __init__(self, services: ServiceContainer = None):
        """Initialize the evaluator."""
        self.services = services
        self.ragas_llm = RagasOpenAI(api_key=SETTINGS.OPENAI_API_KEY)
        self.metrics = [
            faithfulness(llm=self.ragas_llm),
            answer_relevancy(llm=self.ragas_llm),
            context_relevancy(llm=self.ragas_llm),
            context_recall(llm=self.ragas_llm)
        ]

    async def extract_conversations_by_session(
        self, session_id: str, limit: int = 100
    ) -> List[Dict]:
        cursor = self.chat_history_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def extract_conversations_by_customer(
        self, customer_id: str, limit: int = 100
    ) -> Dict[str, List[Dict]]:
        msg_cursor = self.chat_history_collection.find(
            {"customer_id": customer_id}
        ).sort("timestamp", 1).limit(limit)
        all_msg = await msg_cursor.to_list(length=limit)
        return all_msg

    def prepare_conversation_pairs(self, messages: List[Dict]) -> List[Dict]:
        conversation_pairs = []
        current_question = None

        for i, msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role in ["USER", "user"]:
                current_question = content
            elif role in ["BOT"] and current_question and i > 0:
                metadata = msg.get("metadata", {})
                # This is a response to the current question
                conversation_pairs.append({
                    "question": current_question,
                    "answer": content,
                    "metadata": metadata,
                    "timestamp": msg.get("timestamp"),
                    "session_id": msg.get("session_id"),
                    "customer_id": msg.get("customer_id")
                })
                current_question = None
        return conversation_pairs
    
    def extract_context_from_metadata(self, conversation_pairs: List[Dict]) -> List[Dict]:
        for pair in conversation_pairs:
            contexts = []
            metadata = pair.get("metadata", {})
            
            # Extract contexts from various potential metadata fields
            if metadata.get("full_analysis", False):
                # If this includes sentiment analysis results
                contexts.append(f"Sentiment analysis: Score {metadata.get('sentiment_score', 'N/A')}, Confidence: {metadata.get('sentiment_confidence', 'N/A')}")
            
            # Look for course data or search results in your metadata
            if "courses_json" in metadata:
                contexts.append(metadata["courses_json"])
            
            if "search_results" in metadata:
                contexts.append(metadata["search_results"])
            
            # If no contexts could be extracted, use a placeholder
            if not contexts:
                contexts = ["No context metadata available"]
            
            pair["contexts"] = contexts
        
        return conversation_pairs

    def prepare_eval_dataset(self, conversation_pairs: List[Dict]) -> Dataset:
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "session_id": [],
            "timestamp": []
        }
        for pair in conversation_pairs:
            data["question"].append(pair["question"])
            data["answer"].append(pair["answer"])
            data["contexts"].append(
                pair.get("contexts", ["No context available"])
            )
            logger.info(f"MultiTurnSample: {sample}")
            return [sample]
        return []

    def evaluate_multi_turn(
        self, multi_turn_samples: List[MultiTurnSample]
    ) -> Dict[str, float]:
        """Evaluate multi-turn conversations using AspectCritic metrics."""
        if not multi_turn_samples:
            return {}
        results = evaluate(
            dataset=EvaluationDataset(samples=multi_turn_samples),
            metrics=self.multi_turn_metrics
        )
        # Convert DataFrame to dictionary
        results_dict = {}
        for metric in self.multi_turn_metrics:
            metric_name = metric.name if hasattr(metric, 'name') else metric.__class__.__name__
            # Average the scores across all samples
            scores = results.to_pandas()[metric_name].tolist()
            results_dict[metric_name] = sum(scores) / len(scores) if scores else 0
        logger.info(f"Multi-turn evaluation results: {results_dict}")
        return results_dict
    
    async def evaluate_session(
        self, session_id: str, session_chat_limit: int = 100
    ) -> Dict:
        """Evaluate a single conversation session."""
        messages = await self.extract_conversations_by_session(
            session_id, session_chat_limit
        )
        logger.info(f"Messages for session {session_id}: {messages}")
        if not messages:
            return {
                "session_id": session_id,
                "error": "No messages found for this session",
                "metrics": {}
            }
        multi_turn_samples = self.prepare_multi_turn_samples(messages)
        logger.info(f"Multi-turn samples: {multi_turn_samples}")
        multi_turn_metrics = self.evaluate_multi_turn(multi_turn_samples)
        logger.info(f"Multi-turn metrics: {multi_turn_metrics}")
        customer_id = messages[0].get("customer_id") if messages else None
        logger.info(f"Customer ID: {customer_id}")
        result = {
            "session_id": session_id,
            "customer_id": customer_id,
            "message_count": len(messages),
            "metrics": multi_turn_metrics,
            # Add the full conversation thread for reference
            "full_conversation": [
                {
                    "role": msg.get("role", "unknown"),
                    "content": msg.get("content", ""),
                    "timestamp": msg.get("timestamp")
                }
                for msg in messages
            ]
        }
        return result

    async def save_evaluation_results(
        self, results: Dict, base_dir: str = None
    ) -> str:
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results["evaluation_timestamp"] = timestamp
        if not base_dir:
            base_dir = "./data/evaluations/ragas_reports"
            os.makedirs(base_dir, exist_ok=True)
        folder_name = f"session_{results['session_id']}_{timestamp}"
        eval_folder = os.path.join(base_dir, folder_name)
        os.makedirs(eval_folder, exist_ok=True)

        json_filepath = os.path.join(eval_folder, "results.json")
        with open(json_filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        csv_filepath = os.path.join(eval_folder, "results.csv")
        
        # Extract data for CSV
        metrics_data = []
        if "metrics" in results:
            # For single evaluation
            for metric, score in results["metrics"].items():
                metrics_data.append({
                    "id": results.get("session_id", "unknown"),
                    "metric": metric,
                    "score": score,
                    "timestamp": timestamp
                })

        # Write metrics to CSV
        if metrics_data:
            with open(csv_filepath, 'w', newline='') as f:
                fieldnames = ["id", "metric", "score", "timestamp"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in metrics_data:
                    writer.writerow(row)
            
        # Save full conversation threads if available
        if "full_conversation" in results and results["full_conversation"]:
            full_conv_csv_filepath = os.path.join(
                eval_folder, "full_conversation.csv"
            )
            with open(full_conv_csv_filepath, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["id", "role", "content", "timestamp"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, msg in enumerate(results["full_conversation"]):
                    writer.writerow({
                        "id": i + 1,
                        "role": msg.get("role", ""),
                        "content": msg.get("content", ""),
                        "timestamp": msg.get("timestamp", "")
                    })
        
        logger.info(f"Evaluation results saved to: {eval_folder}")
        return eval_folder





# self.single_turn_metrics = [
#             faithfulness(llm=self.ragas_llm),
#             answer_relevancy(llm=self.ragas_llm),
#             context_relevancy(llm=self.ragas_llm),
#             context_recall(llm=self.ragas_llm)
#         ]


# def prepare_conversation_pairs(self, messages: List[Dict]) -> List[Dict]:
#         conversation_pairs = []
#         current_question = None

#         for i, msg in enumerate(messages):  # to get index
#             role = msg.get("role")
#             content = msg.get("content")

#             if role in ["USER", "user"]:
#                 current_question = content
#             elif role in ["BOT", "bot"] and current_question:
#                 metadata = msg.get("metadata", {})
#                 # This is a response to the current question
#                 conversation_pairs.append({
#                     "question": current_question,
#                     "answer": content,
#                     "metadata": metadata,
#                     "top_result": metadata.get('top_result', ''),
#                     "intent": metadata.get('intent', ''),
#                     "timestamp": msg.get("timestamp"),
#                     "session_id": msg.get("session_id"),
#                     "customer_id": msg.get("customer_id")
#                 })
#                 current_question = None
#         return conversation_pairs