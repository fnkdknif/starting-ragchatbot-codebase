"""
Integration tests for RAG system handling content queries
"""
import pytest
from unittest.mock import Mock, patch
from rag_system import RAGSystem


class TestRAGSystemIntegration:
    """Test complete RAG flow for content queries"""

    def test_rag_system_initialization(self, test_config):
        """Test that RAG system initializes all components"""
        rag = RAGSystem(test_config)

        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None

    def test_tools_registered(self, test_config):
        """Test that tools are properly registered"""
        rag = RAGSystem(test_config)

        tool_defs = rag.tool_manager.get_tool_definitions()

        assert len(tool_defs) >= 2  # At least search and outline tools

        # Find search tool
        search_tool = next((t for t in tool_defs if t['name'] == 'search_course_content'), None)
        assert search_tool is not None

        # Find outline tool
        outline_tool = next((t for t in tool_defs if t['name'] == 'get_course_outline'), None)
        assert outline_tool is not None

    def test_add_course_document(self, test_config, tmp_path):
        """Test adding a course document"""
        rag = RAGSystem(test_config)

        # Create a test document
        doc_content = """Course Title: Test Integration Course
Course Link: https://example.com/test
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
This is the introduction to the test course. It covers basic concepts.

Lesson 1: Advanced Topics
Lesson Link: https://example.com/lesson1
This lesson covers advanced topics in the subject matter.
"""
        doc_path = tmp_path / "test_course.txt"
        doc_path.write_text(doc_content)

        # Add document
        course, chunk_count = rag.add_course_document(str(doc_path))

        assert course is not None
        assert course.title == "Test Integration Course"
        assert chunk_count > 0

        # Verify it was added to vector store
        titles = rag.vector_store.get_existing_course_titles()
        assert "Test Integration Course" in titles

    def test_query_with_mocked_ai(self, test_config, mock_course_data):
        """Test complete query flow with mocked AI response"""
        rag = RAGSystem(test_config)

        # Add test data
        course, chunks = mock_course_data
        rag.vector_store.add_course_metadata(course)
        rag.vector_store.add_course_content(chunks)

        # Mock AI response (no tool use)
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Machine Learning is a subset of AI that learns from data."
        mock_text.type = "text"
        mock_response.content = [mock_text]

        with patch.object(rag.ai_generator.client.messages, 'create', return_value=mock_response):
            answer, sources = rag.query("What is machine learning?")

            assert answer is not None
            assert isinstance(answer, str)
            assert len(answer) > 0

    def test_query_with_tool_execution(self, test_config, mock_course_data):
        """Test query flow when AI uses search tool"""
        rag = RAGSystem(test_config)

        # Add test data
        course, chunks = mock_course_data
        rag.vector_store.add_course_metadata(course)
        rag.vector_store.add_course_content(chunks)

        # Mock first response with tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_789"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "machine learning"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [mock_tool_use]

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Based on the course content, Machine Learning is a subset of AI."
        mock_text.type = "text"
        final_response.content = [mock_text]

        with patch.object(rag.ai_generator.client.messages, 'create') as mock_create:
            mock_create.side_effect = [first_response, final_response]

            answer, sources = rag.query("What is machine learning?")

            # Verify response
            assert answer is not None
            assert "Machine Learning" in answer

            # Verify sources were populated
            assert sources is not None
            print(f"\nSources from query: {sources}")

    def test_session_management(self, test_config, mock_course_data):
        """Test that sessions are created and managed"""
        rag = RAGSystem(test_config)

        # Add test data
        course, chunks = mock_course_data
        rag.vector_store.add_course_metadata(course)
        rag.vector_store.add_course_content(chunks)

        # Mock AI response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Test response"
        mock_text.type = "text"
        mock_response.content = [mock_text]

        with patch.object(rag.ai_generator.client.messages, 'create', return_value=mock_response):
            # Create session
            session_id = rag.session_manager.create_session()

            # First query
            answer1, _ = rag.query("What is AI?", session_id)

            # Second query (should have history)
            answer2, _ = rag.query("Tell me more", session_id)

            # Verify history exists
            history = rag.session_manager.get_conversation_history(session_id)
            assert history is not None
            assert "What is AI?" in history

    def test_source_attribution_flow(self, test_config, mock_course_data):
        """Test that sources flow correctly through the system"""
        rag = RAGSystem(test_config)

        # Add test data
        course, chunks = mock_course_data
        rag.vector_store.add_course_metadata(course)
        rag.vector_store.add_course_content(chunks)

        # Mock tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_src"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {
            "query": "neural networks",
            "course_name": "Test Course on AI"
        }

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [mock_tool_use]

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Neural networks are used in deep learning."
        mock_text.type = "text"
        final_response.content = [mock_text]

        with patch.object(rag.ai_generator.client.messages, 'create') as mock_create:
            mock_create.side_effect = [first_response, final_response]

            answer, sources = rag.query("Tell me about neural networks")

            # Sources should be returned
            assert sources is not None
            print(f"\nSources: {sources}")

            # Sources should have expected structure
            if len(sources) > 0:
                assert isinstance(sources[0], dict)
                assert 'text' in sources[0]
                assert 'link' in sources[0]

    def test_get_course_analytics(self, test_config, mock_course_data):
        """Test course analytics retrieval"""
        rag = RAGSystem(test_config)

        # Add test data
        course, chunks = mock_course_data
        rag.vector_store.add_course_metadata(course)

        # Get analytics
        analytics = rag.get_course_analytics()

        assert analytics is not None
        assert 'total_courses' in analytics
        assert 'course_titles' in analytics
        assert analytics['total_courses'] == 1
        assert "Test Course on AI" in analytics['course_titles']

    def test_empty_vector_store_query(self, test_config):
        """Test query behavior with empty vector store"""
        rag = RAGSystem(test_config)

        # Mock AI response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "I don't have information about that in the course materials."
        mock_text.type = "text"
        mock_response.content = [mock_text]

        with patch.object(rag.ai_generator.client.messages, 'create', return_value=mock_response):
            answer, sources = rag.query("What is machine learning?")

            # Should still return a response
            assert answer is not None

    def test_tool_manager_execution(self, test_config, mock_course_data):
        """Test ToolManager executes tools correctly"""
        rag = RAGSystem(test_config)

        # Add test data
        course, chunks = mock_course_data
        rag.vector_store.add_course_metadata(course)
        rag.vector_store.add_course_content(chunks)

        # Execute search tool directly
        result = rag.tool_manager.execute_tool(
            "search_course_content",
            query="machine learning",
            course_name="Test Course on AI"
        )

        assert result is not None
        assert isinstance(result, str)
        print(f"\nDirect tool execution result:\n{result}")

        # Sources should be available
        sources = rag.tool_manager.get_last_sources()
        assert sources is not None
        print(f"\nSources from tool manager: {sources}")

    def test_course_outline_tool_execution(self, test_config, mock_course_data):
        """Test course outline tool through ToolManager"""
        rag = RAGSystem(test_config)

        # Add test data
        course, chunks = mock_course_data
        rag.vector_store.add_course_metadata(course)

        # Execute outline tool
        result = rag.tool_manager.execute_tool(
            "get_course_outline",
            course_name="Test Course"
        )

        assert result is not None
        assert isinstance(result, str)
        assert "Test Course on AI" in result
        assert "Lesson 0" in result
        assert "Lesson 1" in result
        assert "Lesson 2" in result

        print(f"\nOutline tool result:\n{result}")
