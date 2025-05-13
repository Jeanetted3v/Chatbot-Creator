"""Plain Vanilla OpenAI API Wrapper for LLM"""
from src.backend.utils.settings import SETTINGS
from openai import AsyncOpenAI


class LLM():
    def __init__(self):
        self.openai_api_key = SETTINGS.OPENAI_API_KEY
        self.client = AsyncOpenAI(api_key=self.openai_api_key)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = 'gpt-4o-mini',
        temperature: float = 0.2
    ) -> str:
        response = await self.client.chat.completions.create(
           messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
           model=model,
           temperature=temperature,
        #    stream=True
        )
        content = response.choices[0].message.content
        return content
