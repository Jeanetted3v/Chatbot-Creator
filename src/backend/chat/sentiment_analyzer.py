from typing import Tuple, Dict
import logging
import nltk
from vaderSentiment.vaderSentiment import \
    SentimentIntensityAnalyzer as VaderAnalyzer
from src.backend.utils.llm import LLM

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, cfg: Dict):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.vader = VaderAnalyzer()
        self.llm = LLM()
        self.prompts = cfg.sentiment_analyzer_prompts
        self.llm_validate = cfg.sentiment_analyzer.llm_validate_threshold

    def _analyze_vader(self, text: str) -> Tuple[float, float]:
        scores = self.vader.polarity_scores(text)
        
        # Convert compound score from [-1, 1] to [0, 1] range
        score = (scores['compound'] + 1) / 2
        
        # Calculate confidence based on VADER components
        # Higher confidence when pos/neg scores are more polarized
        pos_neg_diff = abs(scores['pos'] - scores['neg'])
        confidence = pos_neg_diff + (1 - scores['neu'])
        confidence = min(max(confidence, 0), 1)  # Ensure 0-1 range
        
        return score, confidence

    async def _validate_with_llm(
        self, text: str, score: float
    ) -> Tuple[float, bool]:
        if not self.llm:
            return score, False
        
        response = await self.llm.generate(
            self.prompts.sys_prompt,
            self.prompts.user_prompt.format(text=text)
        )
        logger.info(f"Response from validate_with_LLM: {response}")
        llm_score = float(response)
            
        # If scores differ significantly, use LLM score
        if abs(llm_score - score) > self.llm_validate:
            return llm_score, True
      
        return score, False

    async def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using vadar and validate with LLM if needed
        
        Returns:
            Dictionary containing:
            - score: Final sentiment score (0-1)
            - confidence: Initial confidence based on VADER components
            - vader_score: Original VADER score
            - llm_validated: Whether LLM validation was used
        """
        try:
            initial_score, confidence = self._analyze_vader(text)
            if confidence < 0.5:
                final_score, was_validated = await self._validate_with_llm(
                    text, initial_score)
                if was_validated:
                    confidence = 0.9  # High confidence after  LLM validation
            else:
                final_score = initial_score
                was_validated = False
            return {
                'score': final_score,
                'confidence': confidence,
                'vader_score': initial_score,
                'llm_validated': was_validated
            }
        except Exception as e:
            raise ValueError(f"Sentiment analysis failed: {str(e)}")
        
    def get_sentiment_label(self, score: float) -> str:
        """Convert numerical score to sentiment label"""
        if score >= 0.75:
            return 'very_positive'
        elif score >= 0.6:
            return 'positive'
        elif score >= 0.4:
            return 'neutral'
        elif score >= 0.25:
            return 'negative'
        else:
            return 'very_negative'