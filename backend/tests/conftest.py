"""
Shared test fixtures for RAG chatbot tests
"""
import pytest
import tempfile
import shutil
import os
import sys
from typing import List, Tuple
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore
from document_processor import DocumentProcessor
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from config import Config
from rag_system import RAGSystem


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
                lesson_link="https://example.com/lesson/0"
            ),
            Lesson(
                lesson_number=1,
                title="Machine Learning Basics",
                lesson_link="https://example.com/lesson/1"
            ),
            Lesson(
                lesson_number=2,
                title="Deep Learning Fundamentals",
                lesson_link="https://example.com/lesson/2"
            )
        ]
    )

    chunks = [
        CourseChunk(
            content="Lesson 0 content: Artificial Intelligence is the simulation of human intelligence by machines. It involves learning, reasoning, and self-correction.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="AI has many applications including natural language processing, computer vision, and robotics.",
            course_title="Test Course on AI",
            lesson_number=0,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Test Course on AI Lesson 1 content: Machine Learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data.",
            course_title="Test Course on AI",
            lesson_number=1,
            chunk_index=2
        ),
        CourseChunk(
            content="Unsupervised learning finds patterns in unlabeled data. Common algorithms include k-means clustering and principal component analysis.",
            course_title="Test Course on AI",
            lesson_number=1,
            chunk_index=3
        ),
        CourseChunk(
            content="Course Test Course on AI Lesson 2 content: Deep Learning uses neural networks with multiple layers. Convolutional neural networks are used for image processing.",
            course_title="Test Course on AI",
            lesson_number=2,
            chunk_index=4
        )
    ]

    return course, chunks


@pytest.fixture
def vector_store_with_data(test_config, mock_course_data):
    """Create a vector store populated with test data"""
    course, chunks = mock_course_data

    store = VectorStore(
        chroma_path=test_config.CHROMA_PATH,
        embedding_model=test_config.EMBEDDING_MODEL,
        max_results=test_config.MAX_RESULTS
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
                "lesson_number": None
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
            self.content = [MockContent("Machine Learning is a subset of AI that enables systems to learn from data.")]
            self.stop_reason = "end_turn"

    return MockResponse()


# API Testing Fixtures

@pytest.fixture
def test_rag_system(test_config, mock_course_data):
    """Create a RAGSystem instance for testing"""
    rag = RAGSystem(test_config)

    # Add test data
    course, chunks = mock_course_data
    rag.vector_store.add_course_metadata(course)
    rag.vector_store.add_course_content(chunks)

    return rag


@pytest.fixture
def test_app(test_rag_system):
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict

    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request/Response models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Optional[str]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = test_rag_system.session_manager.create_session()

            answer, sources = test_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = test_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_system(test_config, mock_course_data):
    """Create a mock RAGSystem for API testing without real AI calls"""
    rag = RAGSystem(test_config)

    # Add test data
    course, chunks = mock_course_data
    rag.vector_store.add_course_metadata(course)
    rag.vector_store.add_course_content(chunks)

    # Mock the query method to avoid real AI calls
    original_query = rag.query

    def mock_query(query: str, session_id: str):
        # Return mock response
        return (
            "This is a mock response about machine learning.",
            [
                {
                    "course_title": "Test Course on AI",
                    "lesson_number": 1,
                    "course_link": "https://example.com/course",
                    "lesson_link": "https://example.com/lesson/1"
                }
            ]
        )

    rag.query = mock_query
    return rag


@pytest.fixture
def mock_app(test_config, mock_course_data, mocker):
    """Create a test FastAPI app with mocked AI generator"""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional, Dict

    # Create RAG system with test data
    rag = RAGSystem(test_config)
    course, chunks = mock_course_data
    rag.vector_store.add_course_metadata(course)
    rag.vector_store.add_course_content(chunks)

    # Mock the AI generator's generate_response method
    def mock_generate_response(query, conversation_history=None, tools=None, tool_manager=None):
        return "This is a mock response about machine learning."

    mocker.patch.object(
        rag.ai_generator,
        'generate_response',
        side_effect=mock_generate_response
    )

    # Also need to mock tool manager sources
    mocker.patch.object(
        rag.tool_manager,
        'get_last_sources',
        return_value=[
            {
                "course_title": "Test Course on AI",
                "lesson_number": "1",
                "course_link": "https://example.com/course",
                "lesson_link": "https://example.com/lesson/1"
            }
        ]
    )

    app = FastAPI(title="Course Materials RAG System - Mock Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Optional[str]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or rag.session_manager.create_session()
            answer, sources = rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def mock_client(mock_app):
    """Create a test client with mocked RAG system"""
    return TestClient(mock_app)
