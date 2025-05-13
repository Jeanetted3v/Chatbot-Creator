"""To run:
python -m src.backend.evaluation.ragas_main
"""
import logging
import hydra
import asyncio
from omegaconf import DictConfig
from src.backend.utils.logging import setup_logging
from src.backend.database.mongodb_client import MongoDBClient
from src.backend.evaluation.ragas import RagasEvaluator
from src.backend.utils.settings import SETTINGS


logger = logging.getLogger(__name__)
logger.info("Setting up logging configuration.")
setup_logging()


async def run_single_session_eval(
    evaluator: RagasEvaluator,
    session_id: str,
    session_chat_limit: int,
    base_dir: str,
    save_results: bool = True
) -> None:
    """Run evaluation for a single session."""
    try:
        logger.info(f"Evaluating session: {session_id}")
        session_result = await evaluator.evaluate_session(
            session_id, session_chat_limit=session_chat_limit
        )
        logger.info(f"Session evaluation result: {session_result}")
        logger.info("Evaluation metrics:")
        for metric, score in session_result.get('metrics', {}).items():
            logger.info(f"  {metric}: {score:.4f}")
        if save_results:
            result_path = await evaluator.save_evaluation_results(
                session_result, base_dir
            )
            logger.info(f"Results saved to: {result_path}")
        return session_result
    except Exception as e:
        logger.error(f"Error during evaluation for session {session_id}: {e}")
        raise


@hydra.main(
    version_base=None,
    config_path="../../../../config",
    config_name="eval")
def main(cfg: DictConfig) -> None:
    async def run_evaluation():
        mongodb_client = None
        try:
            logger.info("Initializing database...")
            mongodb_client = MongoDBClient(SETTINGS.MONGODB_URI)
            await mongodb_client.connect()
            db = mongodb_client.client[cfg.mongodb.db_name]
            chat_history_collection = db[cfg.mongodb.chat_history_collection]
            evaluator = RagasEvaluator(
                chat_history_collection=chat_history_collection
            )
            for session_id in cfg.session_ids:
                await run_single_session_eval(
                    evaluator=evaluator,
                    session_id=session_id,
                    session_chat_limit=cfg.session_chat_limit,
                    base_dir=cfg.ragas_base_dir,
                    save_results=cfg.save_results
                )
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
        finally:
            if mongodb_client:
                await mongodb_client.cleanup()

    asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()