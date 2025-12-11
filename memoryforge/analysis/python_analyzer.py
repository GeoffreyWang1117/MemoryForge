"""Python code analyzer using AST for semantic memory construction."""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from uuid import UUID, uuid4

import structlog

from memoryforge.core.types import SemanticEntity, SemanticRelation

logger = structlog.get_logger()


@dataclass
class FunctionInfo:
    """Information about a function."""

    name: str
    file_path: str
    line_number: int
    end_line: int
    docstring: str | None
    parameters: list[str]
    return_annotation: str | None
    decorators: list[str]
    calls: list[str] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: str | None = None


@dataclass
class ClassInfo:
    """Information about a class."""

    name: str
    file_path: str
    line_number: int
    end_line: int
    docstring: str | None
    bases: list[str]
    decorators: list[str]
    methods: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about an import."""

    module: str
    names: list[str]
    file_path: str
    line_number: int
    is_from_import: bool


class PythonCodeAnalyzer:
    """Analyzes Python code to extract semantic information."""

    def __init__(self):
        self._functions: dict[str, FunctionInfo] = {}
        self._classes: dict[str, ClassInfo] = {}
        self._imports: list[ImportInfo] = []
        self._current_file: str = ""

    def analyze_file(self, file_path: str | Path) -> None:
        """Analyze a single Python file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._current_file = str(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=str(file_path))
            self._visit(tree)
            logger.info(
                "Analyzed file",
                file=str(file_path),
                functions=len(self._functions),
                classes=len(self._classes),
            )
        except SyntaxError as e:
            logger.error("Syntax error in file", file=str(file_path), error=str(e))

    def analyze_directory(self, dir_path: str | Path, pattern: str = "**/*.py") -> None:
        """Analyze all Python files in a directory."""
        dir_path = Path(dir_path)
        for file_path in dir_path.glob(pattern):
            if "__pycache__" not in str(file_path):
                self.analyze_file(file_path)

    def _visit(self, node: ast.AST, class_name: str | None = None) -> None:
        """Visit AST nodes recursively."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                self._process_class(child)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                self._process_function(child, class_name)
            elif isinstance(child, ast.Import):
                self._process_import(child)
            elif isinstance(child, ast.ImportFrom):
                self._process_import_from(child)
            else:
                self._visit(child, class_name)

    def _process_class(self, node: ast.ClassDef) -> None:
        """Process a class definition."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")

        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        class_info = ClassInfo(
            name=node.name,
            file_path=self._current_file,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            bases=bases,
            decorators=decorators,
        )

        # Process methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                class_info.methods.append(item.name)
                self._process_function(item, node.name)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                class_info.attributes.append(item.target.id)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info.attributes.append(target.id)

        key = f"{self._current_file}:{node.name}"
        self._classes[key] = class_info

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str | None = None
    ) -> None:
        """Process a function definition."""
        params = []
        for arg in node.args.args:
            params.append(arg.arg)

        return_annotation = None
        if node.returns:
            return_annotation = ast.unparse(node.returns)

        decorators = [self._get_decorator_name(d) for d in node.decorator_list]

        func_info = FunctionInfo(
            name=node.name,
            file_path=self._current_file,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            docstring=ast.get_docstring(node),
            parameters=params,
            return_annotation=return_annotation,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=class_name is not None,
            class_name=class_name,
        )

        # Find function calls
        func_info.calls = self._extract_calls(node)

        if class_name:
            key = f"{self._current_file}:{class_name}.{node.name}"
        else:
            key = f"{self._current_file}:{node.name}"
        self._functions[key] = func_info

    def _process_import(self, node: ast.Import) -> None:
        """Process an import statement."""
        for alias in node.names:
            self._imports.append(
                ImportInfo(
                    module=alias.name,
                    names=[alias.asname or alias.name],
                    file_path=self._current_file,
                    line_number=node.lineno,
                    is_from_import=False,
                )
            )

    def _process_import_from(self, node: ast.ImportFrom) -> None:
        """Process a from-import statement."""
        module = node.module or ""
        names = [alias.name for alias in node.names]
        self._imports.append(
            ImportInfo(
                module=module,
                names=names,
                file_path=self._current_file,
                line_number=node.lineno,
                is_from_import=True,
            )
        )

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract function calls from a node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(self._get_attribute_name(child.func))
        return calls

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get the name of a decorator."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full name of an attribute access."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    @property
    def functions(self) -> dict[str, FunctionInfo]:
        return self._functions

    @property
    def classes(self) -> dict[str, ClassInfo]:
        return self._classes

    @property
    def imports(self) -> list[ImportInfo]:
        return self._imports

    def _build_entity_map(self) -> dict[str, UUID]:
        """Build a consistent entity ID map."""
        entity_map: dict[str, UUID] = {}
        for key in self._classes:
            entity_map[key] = uuid4()
        for key in self._functions:
            entity_map[key] = uuid4()
        return entity_map

    def to_entities_and_relations(
        self,
    ) -> tuple[list[SemanticEntity], list[SemanticRelation]]:
        """Convert analyzed code to entities and relations with consistent IDs."""
        entity_map = self._build_entity_map()
        entities = list(self._generate_entities(entity_map))
        relations = list(self._generate_relations(entity_map))
        return entities, relations

    def _generate_entities(self, entity_map: dict[str, UUID]) -> Iterator[SemanticEntity]:
        """Generate entities with provided ID map."""
        # Create class entities
        for key, cls in self._classes.items():
            yield SemanticEntity(
                id=entity_map[key],
                name=cls.name,
                entity_type="class",
                properties={
                    "bases": cls.bases,
                    "decorators": cls.decorators,
                    "methods": cls.methods,
                    "attributes": cls.attributes,
                    "docstring": cls.docstring,
                },
                source_file=cls.file_path,
                line_number=cls.line_number,
            )

        # Create function entities
        for key, func in self._functions.items():
            yield SemanticEntity(
                id=entity_map[key],
                name=func.name,
                entity_type="method" if func.is_method else "function",
                properties={
                    "parameters": func.parameters,
                    "return_annotation": func.return_annotation,
                    "decorators": func.decorators,
                    "is_async": func.is_async,
                    "docstring": func.docstring,
                    "class_name": func.class_name,
                    "calls": func.calls,
                },
                source_file=func.file_path,
                line_number=func.line_number,
            )

    def _generate_relations(self, entity_map: dict[str, UUID]) -> Iterator[SemanticRelation]:
        """Generate relations with provided ID map."""
        # Class inheritance relations
        for key, cls in self._classes.items():
            if key not in entity_map:
                continue
            for base in cls.bases:
                for other_key, other_cls in self._classes.items():
                    if other_cls.name == base and other_key in entity_map:
                        yield SemanticRelation(
                            source_id=entity_map[key],
                            target_id=entity_map[other_key],
                            relation_type="extends",
                        )

        # Function call relations
        for key, func in self._functions.items():
            if key not in entity_map:
                continue
            for call in func.calls:
                for other_key, other_func in self._functions.items():
                    if other_func.name == call and other_key in entity_map:
                        yield SemanticRelation(
                            source_id=entity_map[key],
                            target_id=entity_map[other_key],
                            relation_type="calls",
                        )

        # Method belongs to class
        for key, func in self._functions.items():
            if not func.is_method or not func.class_name:
                continue
            class_key = f"{func.file_path}:{func.class_name}"
            if class_key in entity_map and key in entity_map:
                yield SemanticRelation(
                    source_id=entity_map[class_key],
                    target_id=entity_map[key],
                    relation_type="has_method",
                )

    def to_entities(self) -> Iterator[SemanticEntity]:
        """Convert analyzed code to semantic entities."""
        entity_map = self._build_entity_map()

        # Create class entities
        for key, cls in self._classes.items():
            yield SemanticEntity(
                id=entity_map[key],
                name=cls.name,
                entity_type="class",
                properties={
                    "bases": cls.bases,
                    "decorators": cls.decorators,
                    "methods": cls.methods,
                    "attributes": cls.attributes,
                    "docstring": cls.docstring,
                },
                source_file=cls.file_path,
                line_number=cls.line_number,
            )

        # Create function entities
        for key, func in self._functions.items():
            entity_id = uuid4()
            entity_map[key] = entity_id
            yield SemanticEntity(
                id=entity_id,
                name=func.name,
                entity_type="method" if func.is_method else "function",
                properties={
                    "parameters": func.parameters,
                    "return_annotation": func.return_annotation,
                    "decorators": func.decorators,
                    "is_async": func.is_async,
                    "docstring": func.docstring,
                    "class_name": func.class_name,
                },
                source_file=func.file_path,
                line_number=func.line_number,
            )

    def to_relations(self) -> Iterator[SemanticRelation]:
        """Convert analyzed code to semantic relations."""
        entity_map: dict[str, UUID] = {}

        # Build entity map
        for key in self._classes:
            entity_map[key] = uuid4()
        for key in self._functions:
            entity_map[key] = uuid4()

        # Class inheritance relations
        for key, cls in self._classes.items():
            if key not in entity_map:
                continue
            for base in cls.bases:
                # Find base class
                for other_key, other_cls in self._classes.items():
                    if other_cls.name == base and other_key in entity_map:
                        yield SemanticRelation(
                            source_id=entity_map[key],
                            target_id=entity_map[other_key],
                            relation_type="extends",
                        )

        # Function call relations
        for key, func in self._functions.items():
            if key not in entity_map:
                continue
            for call in func.calls:
                # Find called function
                for other_key, other_func in self._functions.items():
                    if other_func.name == call and other_key in entity_map:
                        yield SemanticRelation(
                            source_id=entity_map[key],
                            target_id=entity_map[other_key],
                            relation_type="calls",
                        )

        # Method belongs to class
        for key, func in self._functions.items():
            if not func.is_method or not func.class_name:
                continue
            class_key = f"{func.file_path}:{func.class_name}"
            if class_key in entity_map and key in entity_map:
                yield SemanticRelation(
                    source_id=entity_map[class_key],
                    target_id=entity_map[key],
                    relation_type="has_method",
                )

    def get_summary(self) -> dict:
        """Get a summary of the analyzed code."""
        return {
            "total_files": len(set(f.file_path for f in self._functions.values())),
            "total_classes": len(self._classes),
            "total_functions": len(self._functions),
            "total_imports": len(self._imports),
            "async_functions": sum(1 for f in self._functions.values() if f.is_async),
            "methods": sum(1 for f in self._functions.values() if f.is_method),
        }

    def clear(self) -> None:
        """Clear all analyzed data."""
        self._functions.clear()
        self._classes.clear()
        self._imports.clear()
