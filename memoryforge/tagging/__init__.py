"""Auto-tagging system for memories."""

from memoryforge.tagging.auto_tagger import (
    AutoTagger,
    TaggingConfig,
    TaggingResult,
    TagRule,
)
from memoryforge.tagging.extractors import (
    KeywordExtractor,
    EntityExtractor,
    TopicExtractor,
)

__all__ = [
    "AutoTagger",
    "TaggingConfig",
    "TaggingResult",
    "TagRule",
    "KeywordExtractor",
    "EntityExtractor",
    "TopicExtractor",
]
