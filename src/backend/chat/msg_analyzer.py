"""Handles analysis strategies for chat messages: when to analyse a message,
trigger detection, progressive analysis approach,
Orchestrating the analysis flow.
"""
from typing import Optional
import logging
import re
from src.backend.models.human_agent import AnalysisResult

logger = logging.getLogger(__name__)


class MessageAnalyzer:
    def __init__(self, services):
        self.services = services
        self.cfg = services.cfg.msg_analyzer

    def _check_triggers(self, message: str) -> list[str]:
        """Check for trigger words/patterns in message
        Returns a list of trigger types detected in the message.
        Example: urgency, frustration
        """
        message_lower = message.lower()
        triggered = []
        for trigger_type, pattern in self.cfg.trigger_patterns.items():
            if re.search(pattern, message_lower):
                triggered.append(trigger_type)
                
        return triggered

    def _quick_sentiment_check(self, message: str) -> Optional[float]:
        """Perform quick sentiment check using simple heuristics"""
        message_lower = message.lower()
        if len(message) < self.cfg.min_message_length:  # set to 10 message
            return None
        if any(word in message_lower for word in ['thank', 'good', 'great', 'excellent']):
            return 0.8
        if self._check_triggers(message):
            return 0.3
        return None

    def should_analyze_message(
        self, message: str, message_count: int, last_analyzed_index: int
    ) -> bool:
        """Determine if message should undergo full sentiment analysis"""
        # Skip short messages
        if len(message) < self.cfg.min_message_length:
            return False
        # Always analyze if trigger words are found
        if self._check_triggers(message):
            return True
        # Analyze every Nth message, make sure sentiment is checked regularly
        if (message_count - last_analyzed_index) >= self.cfg.analysis_interval:
            return True
        return False

    async def analyze(
        self,
        message: str,
        message_count: int,
        last_analyzed_index: int
    ) -> AnalysisResult:
        """Orchestrate the analysis of a message"""
        # list of triggers in config file. negative or urgency words.
        triggers = self._check_triggers(message)
        
        # Try quick sentiment check first
        quick_sentiment = self._quick_sentiment_check(message)
        if quick_sentiment is not None:
            return AnalysisResult(
                score=quick_sentiment,
                confidence=0.7,
                method_used='quick_check',
                full_analysis=False,
                triggers_detected=triggers
            )
        
        # Determine if full analysis is needed
        if not self.should_analyze_message(
            message, message_count, last_analyzed_index
        ):
            return AnalysisResult(
                score=0.7,  # Neutral-positive default
                confidence=0.5,
                method_used='skipped',
                full_analysis=False,
                triggers_detected=triggers
            )
        
        # Perform full sentiment analysis
        sentiment_result = await self.services.sentiment_analyzer.analyze_sentiment(message)
        
        return AnalysisResult(
            score=sentiment_result['score'],
            confidence=sentiment_result['confidence'],
            method_used='full_analysis',
            full_analysis=True,
            triggers_detected=triggers,
            analysis_details=sentiment_result
        )