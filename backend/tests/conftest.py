"""
Shared test fixtures for RAG chatbot tests
"""

import os
import shutil
import sys
import tempfile

import pytest

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from models import Course, CourseChunk, Lesson
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_chroma_db):
    """Create a test configuration"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_db
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.ANTHROPIC_API_KEY = "test-api-key"
    return config


@pytest.fixture
def mock_course_data():
    """Create mock course data for testing"""
    course = Course(
        title="Test Course on AI",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction to AI",
                lesson_link="https://example.com/lesson/0",
            ),
            Lesson(
                lesson_number=1,
                title="Machine Learning Basics",
                lesson_link="https://example.com/lesson/1",
            ),
            Lesson(
                lesson_number=2,
                title="Deep Learning Fundamentals",
                lesson_link="https://example.com/lesson/2",
            ),
        ],
    )

    chunks = [
        CourseChunk(
            content="Lesson 0 content: Artificial Intelligence is the simulation of human intelligence by machines. It involves learning, reasoning, and self-correction.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=0,
        ),
        CourseChunk(
            content="AI has many applications including natural language processing, computer vision, and robotics.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=1,
        ),
        CourseChunk(
            content="Course Test Course on AI Lesson 1 content: Machine Learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data.",
            course_title="Test Course on AI",
            lesson_number=1,
            chunk_index=2,
        ),
        CourseChunk(
            content="Unsupervised learning finds patterns in unlabeled data. Common algorithms include k-means clustering and principal component analysis.",
            course_title="Test Course on AI",
            lesson_number=1,
            chunk_index=3,
        ),
        CourseChunk(
            content="Course Test Course on AI Lesson 2 content: Deep Learning uses neural networks with multiple layers. Convolutional neural networks are used for image processing.",
            course_title="Test Course on AI",
            lesson_number=2,
            chunk_index=4,
        ),
    ]

    return course, chunks


@pytest.fixture
def vector_store_with_data(test_config, mock_course_data):
    """Create a vector store populated with test data"""
    course, chunks = mock_course_data

    store = VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS,
    )

    # Add course metadata and content
    store.add_course_metadata(course)
    store.add_course_content(chunks)

    return store


@pytest.fixture
def course_search_tool(vector_store_with_data):
    """Create a CourseSearchTool with populated data"""
    return CourseSearchTool(vector_store_with_data)


@pytest.fixture
def course_outline_tool(vector_store_with_data):
    """Create a CourseOutlineTool with populated data"""
    return CourseOutlineTool(vector_store_with_data)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture
def mock_anthropic_response_no_tools():
    """Mock Anthropic API response without tool use"""

    class MockContent:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class MockResponse:
        def __init__(self):
            self.content = [MockContent("This is a test response from the AI.")]
            self.stop_reason = "end_turn"

    return MockResponse()


@pytest.fixture
def mock_anthropic_response_with_tool():
    """Mock Anthropic API response with tool use"""

    class MockToolContent:
        def __init__(self):
            self.type = "tool_use"
            self.id = "tool_123"
            self.name = "search_course_content"
            self.input = {
                "query": "machine learning",
                "course_name": None,
                "lesson_number": None,
            }

    class MockResponse:
        def __init__(self):
            self.content = [MockToolContent()]
            self.stop_reason = "tool_use"

    return MockResponse()


@pytest.fixture
def mock_anthropic_response_final():
    """Mock final Anthropic API response after tool execution"""

    class MockContent:
        def __init__(self, text):
            self.text = text
            self.type = "text"

    class MockResponse:
        def __init__(self):
            self.content = [
                MockContent(
                    "Machine Learning is a subset of AI that enables systems to learn from data."
                )
            ]
            self.stop_reason = "end_turn"

    return MockResponse()
