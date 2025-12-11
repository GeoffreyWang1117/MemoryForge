"""LLM-based summarization for episodic memory compression."""

import structlog

from memoryforge.core.base import BaseSummarizer

logger = structlog.get_logger()

SUMMARIZE_PROMPT = """Summarize the following conversation concisely, preserving key information, decisions made, and important context. Focus on facts that would be useful for future reference.

Conversation:
{content}

Summary:"""

EXTRACT_FACTS_PROMPT = """Extract the key facts from the following conversation. Return each fact on a new line, starting with a dash (-). Focus on:
- Decisions made
- Technical details mentioned
- User preferences or requirements
- Important outcomes

Conversation:
{content}

Key Facts:"""


class LLMSummarizer(BaseSummarizer):
    """LLM-based summarizer using OpenAI or Anthropic."""

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
    ):
        self._provider = provider
        self._api_key = api_key

        if provider == "openai":
            import openai

            self._client = openai.AsyncOpenAI(api_key=api_key)
            self._model = model or "gpt-4o-mini"
        elif provider == "anthropic":
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._model = model or "claude-3-haiku-20240307"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info("Initialized LLM summarizer", provider=provider, model=self._model)

    async def summarize(self, content: str, max_tokens: int | None = None) -> str:
        """Generate a summary of the content."""
        prompt = SUMMARIZE_PROMPT.format(content=content)

        if self._provider == "openai":
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or 500,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()

        elif self._provider == "anthropic":
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens or 500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()

    async def extract_key_facts(self, content: str) -> list[str]:
        """Extract key facts from content."""
        prompt = EXTRACT_FACTS_PROMPT.format(content=content)

        if self._provider == "openai":
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()

        elif self._provider == "anthropic":
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

        facts = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                facts.append(line[1:].strip())
            elif line:
                facts.append(line)

        return facts
