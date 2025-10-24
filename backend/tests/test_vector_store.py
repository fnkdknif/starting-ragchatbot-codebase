"""
Tests for VectorStore component
"""
import pytest
from vector_store import VectorStore, SearchResults


class TestVectorStore:
    """Test VectorStore operations"""

    def test_vector_store_initialization(self, test_config):
        """Test that vector store initializes correctly"""
        store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        assert store is not None
        assert store.max_results == test_config.MAX_RESULTS
        assert store.course_catalog is not None
        assert store.course_content is not None

    def test_add_course_metadata(self, test_config, mock_course_data):
        """Test adding course metadata to catalog"""
        store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        course, _ = mock_course_data

        # Add course metadata
        store.add_course_metadata(course)

        # Verify it was added
        course_count = store.get_course_count()
        assert course_count == 1

        # Verify course title is retrievable
        titles = store.get_existing_course_titles()
        assert "Test Course on AI" in titles

    def test_add_course_content(self, test_config, mock_course_data):
        """Test adding course content chunks"""
        store = VectorStore(
            chroma_path=test_config.CHROMA_PATH,
            embedding_model=test_config.EMBEDDING_MODEL,
            max_results=test_config.MAX_RESULTS
        )
        _, chunks = mock_course_data

        # Add course content
        store.add_course_content(chunks)

        # Verify content was added by searching
        results = store.course_content.get()
        assert results is not None
        assert len(results['ids']) == len(chunks)

    def test_search_without_filters(self, vector_store_with_data):
        """Test basic search without filters"""
        results = vector_store_with_data.search(query="machine learning")

        assert results is not None
        assert not results.is_empty()
        assert len(results.documents) > 0
        print(f"\nSearch results: {results.documents}")

    def test_search_with_course_filter(self, vector_store_with_data):
        """Test search with course name filter"""
        # Search with full course name
        results = vector_store_with_data.search(
            query="artificial intelligence",
            course_name="Test Course on AI"
        )

        assert not results.is_empty()
        assert len(results.documents) > 0

        # Verify all results are from the correct course
        for meta in results.metadata:
            assert meta['course_title'] == "Test Course on AI"

    def test_search_with_partial_course_name(self, vector_store_with_data):
        """Test search with partial course name (fuzzy matching)"""
        results = vector_store_with_data.search(
            query="machine learning",
            course_name="Test Course"  # Partial name
        )

        assert not results.is_empty()
        print(f"\nPartial course name search results: {results.documents}")

    def test_search_with_lesson_filter(self, vector_store_with_data):
        """Test search with lesson number filter"""
        results = vector_store_with_data.search(
            query="learning",
            lesson_number=1
        )

        assert not results.is_empty()

        # Verify all results are from lesson 1
        for meta in results.metadata:
            assert meta['lesson_number'] == 1

    def test_search_with_both_filters(self, vector_store_with_data):
        """Test search with both course and lesson filters"""
        results = vector_store_with_data.search(
            query="neural networks",
            course_name="Test Course on AI",
            lesson_number=2
        )

        # This should find content from lesson 2
        if not results.is_empty():
            for meta in results.metadata:
                assert meta['course_title'] == "Test Course on AI"
                assert meta['lesson_number'] == 2

    def test_search_no_results(self, vector_store_with_data):
        """Test search that returns no results"""
        results = vector_store_with_data.search(
            query="quantum computing blockchain cryptocurrency"  # Unlikely to match
        )

        # Results might be empty or have very low relevance
        # This is okay - we're testing the handling
        assert results is not None

    def test_search_invalid_course(self, vector_store_with_data):
        """Test search with non-existent course name"""
        results = vector_store_with_data.search(
            query="machine learning",
            course_name="Non-Existent Course XYZ"
        )

        assert results.error is not None
        assert "No course found" in results.error

    def test_resolve_course_name(self, vector_store_with_data):
        """Test course name resolution"""
        # Test exact match
        exact = vector_store_with_data._resolve_course_name("Test Course on AI")
        assert exact == "Test Course on AI"

        # Test partial match
        partial = vector_store_with_data._resolve_course_name("Test Course")
        assert partial == "Test Course on AI"

        # Test non-existent course
        none_result = vector_store_with_data._resolve_course_name("Does Not Exist XYZ")
        assert none_result is None

    def test_get_course_link(self, vector_store_with_data):
        """Test retrieving course link"""
        link = vector_store_with_data.get_course_link("Test Course on AI")
        assert link == "https://example.com/course"

    def test_get_lesson_link(self, vector_store_with_data):
        """Test retrieving lesson link"""
        link = vector_store_with_data.get_lesson_link("Test Course on AI", 1)
        assert link == "https://example.com/lesson/1"

    def test_get_all_courses_metadata(self, vector_store_with_data):
        """Test retrieving all courses metadata"""
        metadata = vector_store_with_data.get_all_courses_metadata()
        assert len(metadata) == 1
        assert metadata[0]['title'] == "Test Course on AI"
        assert 'lessons' in metadata[0]
        assert len(metadata[0]['lessons']) == 3
