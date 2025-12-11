"""Token counting and management utilities."""

import tiktoken


_encoder_cache: dict[str, tiktoken.Encoding] = {}


def get_encoder(model: str = "gpt-4") -> tiktoken.Encoding:
    """Get or create a tiktoken encoder for the specified model."""
    if model not in _encoder_cache:
        try:
            _encoder_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _encoder_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _encoder_cache[model]


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text string."""
    encoder = get_encoder(model)
    return len(encoder.encode(text))


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    model: str = "gpt-4",
    suffix: str = "...",
) -> str:
    """Truncate text to fit within a token limit."""
    encoder = get_encoder(model)
    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return text

    suffix_tokens = encoder.encode(suffix)
    available_tokens = max_tokens - len(suffix_tokens)

    if available_tokens <= 0:
        return suffix

    truncated_tokens = tokens[:available_tokens]
    return encoder.decode(truncated_tokens) + suffix


def estimate_tokens(text: str) -> int:
    """Quick estimation of token count without tiktoken (4 chars ~ 1 token)."""
    return len(text) // 4


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    model: str = "gpt-4",
) -> list[str]:
    """Split text into overlapping chunks by token count."""
    encoder = get_encoder(model)
    tokens = encoder.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(encoder.decode(chunk_tokens))

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks
