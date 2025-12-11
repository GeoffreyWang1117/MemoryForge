"""Tests for the Python code analyzer."""

import pytest
import tempfile
from pathlib import Path

from memoryforge.analysis.python_analyzer import PythonCodeAnalyzer
from memoryforge.analysis.extractor import CodeEntityExtractor


@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return '''
"""Sample module for testing."""

from typing import List, Optional
import os

class BaseHandler:
    """Base class for handlers."""

    def __init__(self, name: str):
        self.name = name

    def handle(self, data: dict) -> bool:
        """Handle data."""
        return True


class UserHandler(BaseHandler):
    """Handler for user operations."""

    def __init__(self, name: str, user_id: int):
        super().__init__(name)
        self.user_id = user_id

    def handle(self, data: dict) -> bool:
        """Handle user data."""
        return self.validate(data)

    def validate(self, data: dict) -> bool:
        """Validate user data."""
        return "user" in data


async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    return {"url": url}


def process_items(items: List[str]) -> int:
    """Process a list of items."""
    count = 0
    for item in items:
        if validate_item(item):
            count += 1
    return count


def validate_item(item: str) -> bool:
    """Validate a single item."""
    return len(item) > 0
'''


@pytest.fixture
def temp_python_file(sample_code):
    """Create a temporary Python file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        return Path(f.name)


def test_analyze_file(temp_python_file):
    """Test analyzing a Python file."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    assert len(analyzer.classes) == 2
    assert len(analyzer.functions) > 0


def test_class_extraction(temp_python_file):
    """Test class extraction."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    # Find BaseHandler
    base_handler = None
    for key, cls in analyzer.classes.items():
        if cls.name == "BaseHandler":
            base_handler = cls
            break

    assert base_handler is not None
    assert base_handler.docstring == "Base class for handlers."
    assert "handle" in base_handler.methods
    assert "__init__" in base_handler.methods


def test_inheritance_detection(temp_python_file):
    """Test inheritance detection."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    # Find UserHandler
    user_handler = None
    for key, cls in analyzer.classes.items():
        if cls.name == "UserHandler":
            user_handler = cls
            break

    assert user_handler is not None
    assert "BaseHandler" in user_handler.bases


def test_function_extraction(temp_python_file):
    """Test function extraction."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    # Find fetch_data
    fetch_data = None
    for key, func in analyzer.functions.items():
        if func.name == "fetch_data":
            fetch_data = func
            break

    assert fetch_data is not None
    assert fetch_data.is_async is True
    assert "url" in fetch_data.parameters
    assert fetch_data.return_annotation == "dict"


def test_function_calls_detection(temp_python_file):
    """Test detection of function calls."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    # Find process_items
    process_items = None
    for key, func in analyzer.functions.items():
        if func.name == "process_items":
            process_items = func
            break

    assert process_items is not None
    assert "validate_item" in process_items.calls


def test_import_extraction(temp_python_file):
    """Test import extraction."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    assert len(analyzer.imports) >= 2

    # Check for typing import
    typing_import = None
    for imp in analyzer.imports:
        if imp.module == "typing" or "typing" in imp.module:
            typing_import = imp
            break

    assert typing_import is not None


def test_method_detection(temp_python_file):
    """Test that methods are correctly identified."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    methods = [f for f in analyzer.functions.values() if f.is_method]
    assert len(methods) > 0

    # Check that validate is a method of UserHandler
    validate_method = None
    for func in methods:
        if func.name == "validate":
            validate_method = func
            break

    assert validate_method is not None
    assert validate_method.class_name == "UserHandler"


def test_entity_generation(temp_python_file):
    """Test entity generation."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    entities = list(analyzer.to_entities())
    assert len(entities) > 0

    # Check entity types
    entity_types = {e.entity_type for e in entities}
    assert "class" in entity_types
    assert "function" in entity_types or "method" in entity_types


def test_relation_generation(temp_python_file):
    """Test relation generation."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    relations = list(analyzer.to_relations())

    # Check for has_method relations
    has_method_relations = [r for r in relations if r.relation_type == "has_method"]
    assert len(has_method_relations) > 0


@pytest.mark.asyncio
async def test_code_entity_extractor(temp_python_file):
    """Test the code entity extractor."""
    extractor = CodeEntityExtractor()
    summary = await extractor.analyze_codebase(temp_python_file)

    assert summary["total_classes"] == 2
    assert summary["total_functions"] > 0
    assert summary["total_entities"] > 0


@pytest.mark.asyncio
async def test_find_callers(temp_python_file):
    """Test finding function callers."""
    extractor = CodeEntityExtractor()
    await extractor.analyze_codebase(temp_python_file)

    callers = extractor.find_callers("validate_item")

    # process_items calls validate_item
    caller_names = [e.name for e in callers]
    assert "process_items" in caller_names


def test_generate_documentation(temp_python_file):
    """Test documentation generation."""
    async def run():
        extractor = CodeEntityExtractor()
        await extractor.analyze_codebase(temp_python_file)
        docs = extractor.generate_documentation()

        assert "# Code Documentation" in docs
        assert "## Classes" in docs
        assert "BaseHandler" in docs
        assert "UserHandler" in docs

    import asyncio
    asyncio.run(run())


def test_get_summary(temp_python_file):
    """Test getting analysis summary."""
    analyzer = PythonCodeAnalyzer()
    analyzer.analyze_file(temp_python_file)

    summary = analyzer.get_summary()

    assert summary["total_files"] == 1
    assert summary["total_classes"] == 2
    assert summary["async_functions"] >= 1


# Cleanup
@pytest.fixture(autouse=True)
def cleanup(temp_python_file):
    """Cleanup temporary files after tests."""
    yield
    if temp_python_file.exists():
        temp_python_file.unlink()
