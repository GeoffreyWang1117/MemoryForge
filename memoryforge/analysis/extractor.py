"""High-level code entity extractor for semantic memory."""

from pathlib import Path
from typing import Iterator

import structlog

from memoryforge.analysis.python_analyzer import PythonCodeAnalyzer
from memoryforge.core.types import SemanticEntity, SemanticRelation
from memoryforge.memory.semantic.memory import SemanticMemory

logger = structlog.get_logger()


class CodeEntityExtractor:
    """Extracts code entities and builds semantic memory."""

    def __init__(self, semantic_memory: SemanticMemory | None = None):
        self._analyzer = PythonCodeAnalyzer()
        self._semantic_memory = semantic_memory
        self._entities: list[SemanticEntity] = []
        self._relations: list[SemanticRelation] = []

    async def analyze_codebase(self, path: str | Path) -> dict:
        """Analyze a codebase and extract entities.

        Args:
            path: Path to file or directory to analyze

        Returns:
            Summary of extracted entities
        """
        path = Path(path)

        if path.is_file():
            self._analyzer.analyze_file(path)
        elif path.is_dir():
            self._analyzer.analyze_directory(path)
        else:
            raise ValueError(f"Invalid path: {path}")

        # Convert to entities and relations (with consistent IDs)
        self._entities, self._relations = self._analyzer.to_entities_and_relations()

        # Store in semantic memory if available
        if self._semantic_memory:
            for entity in self._entities:
                await self._semantic_memory.add_entity(entity)
            for relation in self._relations:
                await self._semantic_memory.add_relation(relation)

            logger.info(
                "Stored entities in semantic memory",
                entities=len(self._entities),
                relations=len(self._relations),
            )

        return self.get_summary()

    def get_summary(self) -> dict:
        """Get summary of extracted entities."""
        analyzer_summary = self._analyzer.get_summary()

        entity_types = {}
        for entity in self._entities:
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

        relation_types = {}
        for relation in self._relations:
            relation_types[relation.relation_type] = (
                relation_types.get(relation.relation_type, 0) + 1
            )

        return {
            **analyzer_summary,
            "entity_types": entity_types,
            "relation_types": relation_types,
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
        }

    def get_entities(self) -> list[SemanticEntity]:
        """Get all extracted entities."""
        return self._entities

    def get_relations(self) -> list[SemanticRelation]:
        """Get all extracted relations."""
        return self._relations

    def get_entity_by_name(self, name: str) -> list[SemanticEntity]:
        """Find entities by name."""
        return [e for e in self._entities if e.name == name]

    def get_entities_by_type(self, entity_type: str) -> list[SemanticEntity]:
        """Get all entities of a given type."""
        return [e for e in self._entities if e.entity_type == entity_type]

    def get_entity_relations(self, entity_id) -> list[SemanticRelation]:
        """Get all relations for an entity."""
        return [
            r
            for r in self._relations
            if r.source_id == entity_id or r.target_id == entity_id
        ]

    def find_callers(self, function_name: str) -> list[SemanticEntity]:
        """Find all functions that call a given function."""
        target_entities = self.get_entity_by_name(function_name)
        if not target_entities:
            return []

        target_ids = {e.id for e in target_entities}
        caller_ids = set()

        for relation in self._relations:
            if relation.relation_type == "calls" and relation.target_id in target_ids:
                caller_ids.add(relation.source_id)

        return [e for e in self._entities if e.id in caller_ids]

    def find_callees(self, function_name: str) -> list[SemanticEntity]:
        """Find all functions called by a given function."""
        source_entities = self.get_entity_by_name(function_name)
        if not source_entities:
            return []

        source_ids = {e.id for e in source_entities}
        callee_ids = set()

        for relation in self._relations:
            if relation.relation_type == "calls" and relation.source_id in source_ids:
                callee_ids.add(relation.target_id)

        return [e for e in self._entities if e.id in callee_ids]

    def get_class_hierarchy(self, class_name: str) -> dict:
        """Get inheritance hierarchy for a class."""
        class_entities = [
            e for e in self._entities if e.name == class_name and e.entity_type == "class"
        ]
        if not class_entities:
            return {}

        def get_bases(entity_id) -> list[dict]:
            bases = []
            for relation in self._relations:
                if relation.relation_type == "extends" and relation.source_id == entity_id:
                    base_entity = next(
                        (e for e in self._entities if e.id == relation.target_id), None
                    )
                    if base_entity:
                        bases.append({
                            "name": base_entity.name,
                            "id": str(base_entity.id),
                            "bases": get_bases(base_entity.id),
                        })
            return bases

        def get_subclasses(entity_id) -> list[dict]:
            subclasses = []
            for relation in self._relations:
                if relation.relation_type == "extends" and relation.target_id == entity_id:
                    sub_entity = next(
                        (e for e in self._entities if e.id == relation.source_id), None
                    )
                    if sub_entity:
                        subclasses.append({
                            "name": sub_entity.name,
                            "id": str(sub_entity.id),
                            "subclasses": get_subclasses(sub_entity.id),
                        })
            return subclasses

        entity = class_entities[0]
        return {
            "name": entity.name,
            "id": str(entity.id),
            "bases": get_bases(entity.id),
            "subclasses": get_subclasses(entity.id),
        }

    def generate_documentation(self) -> str:
        """Generate markdown documentation from extracted entities."""
        lines = ["# Code Documentation\n"]

        # Classes
        classes = self.get_entities_by_type("class")
        if classes:
            lines.append("## Classes\n")
            for cls in sorted(classes, key=lambda e: e.name):
                lines.append(f"### {cls.name}\n")
                lines.append(f"**File:** `{cls.source_file}:{cls.line_number}`\n")

                if cls.properties.get("docstring"):
                    lines.append(f"\n{cls.properties['docstring']}\n")

                if cls.properties.get("bases"):
                    bases = ", ".join(cls.properties["bases"])
                    lines.append(f"\n**Inherits from:** {bases}\n")

                if cls.properties.get("methods"):
                    lines.append("\n**Methods:**\n")
                    for method in cls.properties["methods"]:
                        lines.append(f"- `{method}`\n")

                lines.append("\n---\n")

        # Functions
        functions = self.get_entities_by_type("function")
        if functions:
            lines.append("## Functions\n")
            for func in sorted(functions, key=lambda e: e.name):
                params = ", ".join(func.properties.get("parameters", []))
                ret = func.properties.get("return_annotation", "")
                sig = f"{func.name}({params})"
                if ret:
                    sig += f" -> {ret}"

                lines.append(f"### `{sig}`\n")
                lines.append(f"**File:** `{func.source_file}:{func.line_number}`\n")

                if func.properties.get("is_async"):
                    lines.append("**async**\n")

                if func.properties.get("docstring"):
                    lines.append(f"\n{func.properties['docstring']}\n")

                lines.append("\n---\n")

        return "".join(lines)

    def clear(self) -> None:
        """Clear all extracted data."""
        self._analyzer.clear()
        self._entities.clear()
        self._relations.clear()
