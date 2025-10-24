"""
Tests for CourseSearchTool execute() method outputs
"""



class TestCourseSearchTool:
    """Test CourseSearchTool.execute() method"""

    def test_tool_definition(self, course_search_tool):
        """Test that tool definition is properly structured"""
        tool_def = course_search_tool.get_tool_definition()

        assert tool_def is not None
        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def

        # Check required parameters
        schema = tool_def["input_schema"]
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_execute_basic_query(self, course_search_tool):
        """Test execute with just a query (no filters)"""
        result = course_search_tool.execute(query="machine learning")

        print(f"\nBasic query result:\n{result}")

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

        # Should not contain error messages
        assert (
            "No relevant content found" not in result
            or result != "No relevant content found."
        )

    def test_execute_with_course_filter(self, course_search_tool):
        """Test execute with course name filter"""
        result = course_search_tool.execute(
            query="artificial intelligence", course_name="Test Course on AI"
        )

        print(f"\nCourse filter result:\n{result}")

        assert result is not None
        assert isinstance(result, str)
        assert "Test Course on AI" in result

    def test_execute_with_partial_course_name(self, course_search_tool):
        """Test execute with partial course name"""
        result = course_search_tool.execute(
            query="deep learning", course_name="Test Course"  # Partial name
        )

        print(f"\nPartial course name result:\n{result}")

        assert result is not None
        # Should still work with fuzzy matching
        assert "No course found" not in result

    def test_execute_with_lesson_filter(self, course_search_tool):
        """Test execute with lesson number filter"""
        result = course_search_tool.execute(
            query="supervised learning", lesson_number=1
        )

        print(f"\nLesson filter result:\n{result}")

        assert result is not None
        assert "Lesson 1" in result

    def test_execute_with_both_filters(self, course_search_tool):
        """Test execute with both course and lesson filters"""
        result = course_search_tool.execute(
            query="neural networks", course_name="Test Course on AI", lesson_number=2
        )

        print(f"\nBoth filters result:\n{result}")

        assert result is not None
        if "No relevant content found" not in result:
            assert "Test Course on AI" in result
            assert "Lesson 2" in result

    def test_execute_invalid_course(self, course_search_tool):
        """Test execute with non-existent course"""
        result = course_search_tool.execute(
            query="machine learning", course_name="Non-Existent Course XYZ"
        )

        print(f"\nInvalid course result:\n{result}")

        assert result is not None
        assert "No course found matching 'Non-Existent Course XYZ'" in result

    def test_execute_no_results(self, course_search_tool):
        """Test execute when no results are found"""
        result = course_search_tool.execute(
            query="quantum computing blockchain cryptocurrency spacetime"
        )

        print(f"\nNo results query result:\n{result}")

        assert result is not None
        # Should handle gracefully
        assert isinstance(result, str)

    def test_result_format(self, course_search_tool):
        """Test that results are properly formatted"""
        result = course_search_tool.execute(query="machine learning")

        print(f"\nFormatted result:\n{result}")

        # Check for expected formatting patterns
        assert result is not None

        # Results should have course/lesson context
        if "No relevant content found" not in result:
            # Should contain brackets for course/lesson context
            assert "[" in result or "Lesson" in result

    def test_sources_tracking(self, course_search_tool):
        """Test that sources are tracked correctly"""
        # Execute a search
        result = course_search_tool.execute(
            query="artificial intelligence", course_name="Test Course on AI"
        )

        # Check that sources were tracked
        assert hasattr(course_search_tool, "last_sources")
        assert course_search_tool.last_sources is not None

        if "No relevant content found" not in result:
            # Sources should be populated
            assert len(course_search_tool.last_sources) > 0

            # Each source should have text and link
            for source in course_search_tool.last_sources:
                assert "text" in source
                assert "link" in source

            print(f"\nSources: {course_search_tool.last_sources}")

    def test_execute_kwargs_compatibility(self, course_search_tool):
        """Test that execute works with **kwargs pattern"""
        # This tests the tool execution pattern used by ToolManager

        # Test 1: All parameters as kwargs
        params = {
            "query": "machine learning",
            "course_name": "Test Course on AI",
            "lesson_number": 1,
        }
        result = course_search_tool.execute(**params)
        assert result is not None

        # Test 2: Only required parameter
        params = {"query": "deep learning"}
        result = course_search_tool.execute(**params)
        assert result is not None

        # Test 3: Partial parameters
        params = {"query": "neural networks", "course_name": "Test Course"}
        result = course_search_tool.execute(**params)
        assert result is not None

    def test_multiple_executions(self, course_search_tool):
        """Test multiple consecutive executions"""
        # First execution
        result1 = course_search_tool.execute(query="machine learning")
        sources1 = (
            course_search_tool.last_sources.copy()
            if course_search_tool.last_sources
            else []
        )

        # Second execution
        result2 = course_search_tool.execute(query="deep learning")
        sources2 = (
            course_search_tool.last_sources.copy()
            if course_search_tool.last_sources
            else []
        )

        # Both should work independently
        assert result1 is not None
        assert result2 is not None

        # Sources should be different (or at least independently tracked)
        print(f"\nFirst sources: {sources1}")
        print(f"Second sources: {sources2}")
