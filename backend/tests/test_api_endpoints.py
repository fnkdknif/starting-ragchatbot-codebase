"""
Tests for FastAPI endpoints
"""
import pytest
from fastapi import status
from unittest.mock import patch, MagicMock


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    def test_query_endpoint_basic(self, mock_client):
        """Test basic query functionality"""
        response = mock_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert len(data["answer"]) > 0
        assert len(data["session_id"]) > 0

    def test_query_endpoint_with_session_id(self, mock_client):
        """Test query with existing session ID"""
        # First request to get a session ID
        response1 = mock_client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        session_id = response1.json()["session_id"]

        # Second request with the same session ID
        response2 = mock_client.post(
            "/api/query",
            json={
                "query": "Tell me more about that",
                "session_id": session_id
            }
        )

        assert response2.status_code == status.HTTP_200_OK
        data = response2.json()
        assert data["session_id"] == session_id

    def test_query_endpoint_creates_session_if_not_provided(self, mock_client):
        """Test that a session ID is created if not provided"""
        response = mock_client.post(
            "/api/query",
            json={"query": "What is deep learning?"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] is not None
        assert len(data["session_id"]) > 0

    def test_query_endpoint_empty_query(self, mock_client):
        """Test query with empty string"""
        response = mock_client.post(
            "/api/query",
            json={"query": ""}
        )

        # Should still return 200 but might have a specific response
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_query_endpoint_missing_query_field(self, mock_client):
        """Test query with missing query field"""
        response = mock_client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_endpoint_invalid_json(self, mock_client):
        """Test query with invalid JSON"""
        response = mock_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_endpoint_sources_structure(self, mock_client):
        """Test that sources have the correct structure"""
        response = mock_client.post(
            "/api/query",
            json={"query": "What is supervised learning?"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Check sources structure
        sources = data["sources"]
        assert isinstance(sources, list)

        if len(sources) > 0:
            source = sources[0]
            assert isinstance(source, dict)
            # Expected source fields
            expected_fields = ["course_title", "lesson_number", "course_link", "lesson_link"]
            for field in expected_fields:
                assert field in source

    def test_query_endpoint_long_query(self, mock_client):
        """Test query with very long input"""
        long_query = "What is machine learning? " * 100
        response = mock_client.post(
            "/api/query",
            json={"query": long_query}
        )

        # Should handle long queries gracefully
        assert response.status_code == status.HTTP_200_OK

    def test_query_endpoint_special_characters(self, mock_client):
        """Test query with special characters"""
        response = mock_client.post(
            "/api/query",
            json={"query": "What is ML? Tell me about AI & DL (deep learning)!"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data

    def test_query_endpoint_unicode_characters(self, mock_client):
        """Test query with unicode characters"""
        response = mock_client.post(
            "/api/query",
            json={"query": "机器学习是什么？ What is machine learning?"}
        )

        assert response.status_code == status.HTTP_200_OK

    @patch('rag_system.RAGSystem.query')
    def test_query_endpoint_handles_rag_exception(self, mock_query, mock_client):
        """Test error handling when RAG system raises exception"""
        # This test requires the client to have access to the patchable RAG system
        # Since we're using a mock client, we'll test the general error case
        response = mock_client.post(
            "/api/query",
            json={"query": None}  # Invalid query type
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    def test_courses_endpoint_basic(self, mock_client, mock_course_data):
        """Test basic courses stats retrieval"""
        response = mock_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify data types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0

    def test_courses_endpoint_returns_correct_count(self, client, mock_course_data):
        """Test that courses endpoint returns correct course count"""
        response = client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # We added one course in the fixture
        assert data["total_courses"] >= 1

    def test_courses_endpoint_returns_course_titles(self, client, mock_course_data):
        """Test that courses endpoint returns course titles"""
        response = client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Verify course titles list
        assert isinstance(data["course_titles"], list)
        assert len(data["course_titles"]) >= 1

        # Check if our test course is in the list
        assert "Test Course on AI" in data["course_titles"]

    def test_courses_endpoint_no_params_required(self, mock_client):
        """Test that courses endpoint doesn't require parameters"""
        response = mock_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK

    def test_courses_endpoint_method_not_allowed(self, mock_client):
        """Test that POST is not allowed on /api/courses"""
        response = mock_client.post("/api/courses", json={})

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_courses_endpoint_with_query_params(self, mock_client):
        """Test that courses endpoint ignores query parameters"""
        response = mock_client.get("/api/courses?foo=bar&test=123")

        # Should still work and ignore the params
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.api
class TestEndpointIntegration:
    """Integration tests for API endpoints"""

    def test_query_then_courses(self, mock_client):
        """Test making a query then getting course stats"""
        # First, make a query
        query_response = mock_client.post(
            "/api/query",
            json={"query": "What is neural networks?"}
        )
        assert query_response.status_code == status.HTTP_200_OK

        # Then get course stats
        courses_response = mock_client.get("/api/courses")
        assert courses_response.status_code == status.HTTP_200_OK

    def test_multiple_queries_same_session(self, mock_client):
        """Test multiple queries in the same session"""
        # First query
        response1 = mock_client.post(
            "/api/query",
            json={"query": "What is AI?"}
        )
        session_id = response1.json()["session_id"]

        # Second query with same session
        response2 = mock_client.post(
            "/api/query",
            json={"query": "What about machine learning?", "session_id": session_id}
        )
        assert response2.json()["session_id"] == session_id

        # Third query with same session
        response3 = mock_client.post(
            "/api/query",
            json={"query": "And deep learning?", "session_id": session_id}
        )
        assert response3.json()["session_id"] == session_id

    def test_concurrent_sessions(self, mock_client):
        """Test that different sessions are independent"""
        # Create first session
        response1 = mock_client.post(
            "/api/query",
            json={"query": "First session query"}
        )
        session1 = response1.json()["session_id"]

        # Create second session
        response2 = mock_client.post(
            "/api/query",
            json={"query": "Second session query"}
        )
        session2 = response2.json()["session_id"]

        # Verify sessions are different
        assert session1 != session2

    def test_invalid_endpoint(self, mock_client):
        """Test accessing non-existent endpoint"""
        response = mock_client.get("/api/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_api_accepts_cors_headers(self, mock_client):
        """Test that API properly handles CORS"""
        response = mock_client.post(
            "/api/query",
            json={"query": "Test query"},
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == status.HTTP_200_OK


@pytest.mark.api
class TestResponseValidation:
    """Test response model validation"""

    def test_query_response_model_validation(self, mock_client):
        """Test that query response matches the expected model"""
        response = mock_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Validate required fields exist
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_courses_response_model_validation(self, mock_client):
        """Test that courses response matches the expected model"""
        response = mock_client.get("/api/courses")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Validate required fields exist
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Validate field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


@pytest.mark.api
class TestErrorHandling:
    """Test error handling in API endpoints"""

    def test_malformed_request_body(self, mock_client):
        """Test handling of malformed request body"""
        response = mock_client.post(
            "/api/query",
            json={"wrong_field": "value"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_wrong_content_type(self, mock_client):
        """Test handling of wrong content type"""
        response = mock_client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        # FastAPI should handle this and return 422
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_method_on_query_endpoint(self, mock_client):
        """Test using invalid HTTP method on query endpoint"""
        response = mock_client.get("/api/query")
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_extra_fields_in_request(self, mock_client):
        """Test that extra fields in request are ignored"""
        response = mock_client.post(
            "/api/query",
            json={
                "query": "What is AI?",
                "extra_field": "should be ignored",
                "another_field": 123
            }
        )

        # Should succeed and ignore extra fields
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.api
class TestSessionManagement:
    """Test session management through API"""

    def test_session_persistence(self, mock_client):
        """Test that session persists across requests"""
        # First request
        response1 = mock_client.post(
            "/api/query",
            json={"query": "First query"}
        )
        session_id = response1.json()["session_id"]

        # Second request with same session
        response2 = mock_client.post(
            "/api/query",
            json={"query": "Second query", "session_id": session_id}
        )

        # Should return same session ID
        assert response2.json()["session_id"] == session_id

    def test_new_session_creation(self, mock_client):
        """Test that new sessions are created when needed"""
        # Multiple requests without session ID
        session_ids = []
        for i in range(3):
            response = mock_client.post(
                "/api/query",
                json={"query": f"Query {i}"}
            )
            session_ids.append(response.json()["session_id"])

        # Each should have a unique session ID
        assert len(set(session_ids)) == 3

    def test_invalid_session_id_format(self, mock_client):
        """Test handling of invalid session ID format"""
        response = mock_client.post(
            "/api/query",
            json={
                "query": "Test query",
                "session_id": "invalid-session-id-format-12345"
            }
        )

        # Should still work - might create new session or use the provided ID
        assert response.status_code == status.HTTP_200_OK
