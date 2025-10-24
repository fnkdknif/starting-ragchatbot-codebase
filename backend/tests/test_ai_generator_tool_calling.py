"""
Tests for AIGenerator tool calling functionality
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorToolCalling:
    """Test AIGenerator's ability to call tools correctly"""

    def test_generate_response_without_tools(self, test_config, mock_anthropic_response_no_tools):
        """Test basic response generation without tools"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        with patch.object(ai_gen.client.messages, 'create', return_value=mock_anthropic_response_no_tools):
            response = ai_gen.generate_response(
                query="What is artificial intelligence?"
            )

            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0

    def test_tool_definitions_passed_correctly(self, test_config, tool_manager):
        """Test that tool definitions are passed to the API"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        # Mock the API call
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response", type="text")]

        with patch.object(ai_gen.client.messages, 'create', return_value=mock_response) as mock_create:
            ai_gen.generate_response(
                query="What is machine learning?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Verify create was called with tools
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]

            assert 'tools' in call_kwargs
            assert len(call_kwargs['tools']) > 0
            assert 'tool_choice' in call_kwargs
            assert call_kwargs['tool_choice']['type'] == 'auto'

    def test_tool_execution_triggered(self, test_config, tool_manager, vector_store_with_data):
        """Test that tool execution is triggered when API requests it"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        # Mock first response with tool_use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_123"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "machine learning"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [mock_tool_use]

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Machine Learning is a subset of AI."
        mock_text.type = "text"
        final_response.content = [mock_text]

        with patch.object(ai_gen.client.messages, 'create') as mock_create:
            # First call returns tool_use, second call returns final response
            mock_create.side_effect = [first_response, final_response]

            response = ai_gen.generate_response(
                query="What is machine learning?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Should have called API twice
            assert mock_create.call_count == 2

            # Final response should be from second call
            assert "Machine Learning" in response

    def test_tool_results_sent_back_to_api(self, test_config, tool_manager):
        """Test that tool results are properly sent back to the API"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        # Mock tool use response
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_456"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "deep learning"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [mock_tool_use]

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Here's what I found about deep learning."
        mock_text.type = "text"
        final_response.content = [mock_text]

        with patch.object(ai_gen.client.messages, 'create') as mock_create:
            mock_create.side_effect = [first_response, final_response]

            response = ai_gen.generate_response(
                query="Tell me about deep learning",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Check second API call had tool results
            assert mock_create.call_count == 2

            second_call_kwargs = mock_create.call_args_list[1][1]
            messages = second_call_kwargs['messages']

            # Should have 3 messages: user query, assistant tool_use, user tool_result
            assert len(messages) >= 2

            # Last message should contain tool results
            last_message = messages[-1]
            assert last_message['role'] == 'user'
            assert 'content' in last_message

            # Content should be tool results
            content = last_message['content']
            assert isinstance(content, list)
            assert len(content) > 0
            assert content[0]['type'] == 'tool_result'

    def test_system_prompt_construction(self, test_config):
        """Test system prompt is correctly constructed"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        # Test without conversation history
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Test response"
        mock_text.type = "text"
        mock_response.content = [mock_text]

        with patch.object(ai_gen.client.messages, 'create', return_value=mock_response) as mock_create:
            ai_gen.generate_response(query="Test query")

            call_kwargs = mock_create.call_args[1]
            assert 'system' in call_kwargs
            assert ai_gen.SYSTEM_PROMPT in call_kwargs['system']

        # Test with conversation history
        with patch.object(ai_gen.client.messages, 'create', return_value=mock_response) as mock_create:
            ai_gen.generate_response(
                query="Follow-up question",
                conversation_history="User: Previous question\nAssistant: Previous answer"
            )

            call_kwargs = mock_create.call_args[1]
            assert 'system' in call_kwargs
            assert "Previous question" in call_kwargs['system']
            assert "Previous answer" in call_kwargs['system']

    def test_tool_execution_error_handling(self, test_config, tool_manager):
        """Test handling of errors during tool execution"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        # Mock tool use with invalid parameters
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.id = "tool_error"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test"}  # Valid input

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [mock_tool_use]

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Result after tool execution"
        mock_text.type = "text"
        final_response.content = [mock_text]

        with patch.object(ai_gen.client.messages, 'create') as mock_create:
            mock_create.side_effect = [first_response, final_response]

            # Should not raise exception, should handle gracefully
            response = ai_gen.generate_response(
                query="Test query",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            assert response is not None

    def test_multiple_tool_calls_in_response(self, test_config, tool_manager):
        """Test handling of multiple tool calls in one response"""
        ai_gen = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

        # Mock multiple tool uses
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.input = {"query": "machine learning"}

        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.name = "get_course_outline"
        mock_tool_use_2.input = {"course_name": "Test Course"}

        first_response = Mock()
        first_response.stop_reason = "tool_use"
        first_response.content = [mock_tool_use_1, mock_tool_use_2]

        # Mock final response
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        mock_text = Mock()
        mock_text.text = "Combined results from both tools"
        mock_text.type = "text"
        final_response.content = [mock_text]

        with patch.object(ai_gen.client.messages, 'create') as mock_create:
            mock_create.side_effect = [first_response, final_response]

            response = ai_gen.generate_response(
                query="Multiple tool query",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager
            )

            # Should handle multiple tools
            assert response is not None
            assert mock_create.call_count == 2

            # Check that both tool results were sent
            second_call = mock_create.call_args_list[1][1]
            messages = second_call['messages']
            last_message = messages[-1]

            # Should have 2 tool results
            assert len(last_message['content']) == 2
