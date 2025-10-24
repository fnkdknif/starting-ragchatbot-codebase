# Test Results and Proposed Fixes

## Test Summary

**Total Tests**: 44
**Passed**: 41
**Failed**: 3

## Test Failures Analysis

### 1. Critical Bug: Course Name Resolution Always Returns a Match

**Failing Tests**:
- `test_vector_store.py::test_resolve_course_name`
- `test_vector_store.py::test_search_invalid_course`
- `test_course_search_tool.py::test_execute_invalid_course`

**Root Cause**:
The `_resolve_course_name()` method in `vector_store.py` (line 102-116) uses semantic vector search which **always returns the closest match**, even when the query is completely unrelated to any course name.

**Example**:
- Query: "Does Not Exist XYZ"
- Expected: `None` (no match)
- Actual: "Test Course on AI" (closest semantic match)

This means the system will **never reject invalid course names** and will always search content from some course, even when the user specifies a non-existent course.

**Impact**:
- Users asking about "Non-Existent Course XYZ" will get results from a random course
- Search results will be misleading and incorrect
- No way to tell users "that course doesn't exist"

---

## Proposed Fixes

### Fix #1: Add Similarity Threshold to Course Name Resolution

**Location**: `backend/vector_store.py`, line 102-116

**Current Code**:
```python
def _resolve_course_name(self, course_name: str) -> Optional[str]:
    """Use vector search to find best matching course by name"""
    try:
        results = self.course_catalog.query(
            query_texts=[course_name],
            n_results=1
        )

        if results['documents'][0] and results['metadatas'][0]:
            # Return the title (which is now the ID)
            return results['metadatas'][0][0]['title']
    except Exception as e:
        print(f"Error resolving course name: {e}")

    return None
```

**Proposed Fix**:
```python
def _resolve_course_name(self, course_name: str) -> Optional[str]:
    """Use vector search to find best matching course by name"""
    try:
        results = self.course_catalog.query(
            query_texts=[course_name],
            n_results=1
        )

        if results['documents'][0] and results['metadatas'][0] and results['distances'][0]:
            # Check similarity threshold - only accept if distance is reasonable
            # Lower distance = better match. Typical good matches are < 0.5
            # We use 0.8 as threshold to allow some fuzzy matching but reject nonsense
            distance = results['distances'][0][0]

            if distance < 0.8:  # Configurable threshold
                return results['metadatas'][0][0]['title']
            else:
                # Distance too high - no good match found
                return None

    except Exception as e:
        print(f"Error resolving course name: {e}")

    return None
```

**Rationale**:
- Semantic search returns a distance score (lower = better match)
- By checking if distance < threshold, we can reject poor matches
- Threshold of 0.8 allows fuzzy matching ("MCP" → "Introduction to MCP") but rejects nonsense
- Makes the system more robust and user-friendly

---

### Fix #2: Add Configuration for Similarity Threshold

**Location**: `backend/config.py`

**Add Configuration**:
```python
@dataclass
class Config:
    # ... existing config ...

    # Course name matching threshold
    COURSE_NAME_SIMILARITY_THRESHOLD: float = 0.8  # Max distance for valid match
```

**Update VectorStore Constructor**:
```python
class VectorStore:
    def __init__(self, chroma_path: str, embedding_model: str, max_results: int = 5,
                 course_name_threshold: float = 0.8):
        self.max_results = max_results
        self.course_name_threshold = course_name_threshold
        # ... rest of init ...
```

---

### Fix #3: Improve Error Messages

**Location**: `backend/search_tools.py`, CourseSearchTool.execute()

The current implementation already returns good error messages, but we should ensure they're informative:

**Current behavior works correctly** after Fix #1 is applied:
- When `course_name` doesn't match anything, `_resolve_course_name` returns `None`
- This triggers the error message: `"No course found matching '{course_name}'"`

No changes needed here once Fix #1 is implemented.

---

## Additional Findings from Tests

### ✅ Working Correctly:

1. **VectorStore Operations**: All basic operations work (add metadata, add content, search)
2. **CourseSearchTool**: Execute method works with all parameter combinations
3. **AIGenerator Tool Calling**: Correctly handles tool execution flow
4. **RAG System Integration**: All components integrate properly
5. **Session Management**: Conversation history tracking works
6. **Source Attribution**: Sources flow correctly through the system
7. **Tool Manager**: Executes both search and outline tools correctly

### ⚠️ Areas of Concern (Not Failures, But Worth Noting):

1. **Semantic Search Sensitivity**: Even with threshold, very generic queries might match courses unexpectedly
2. **API Credit Issue**: The actual deployment has API credit issues (seen in logs), but tests mock this successfully
3. **No Input Validation**: Tool inputs aren't validated (e.g., negative lesson numbers)

---

## Implementation Priority

### High Priority (Required):
1. ✅ **Fix #1**: Add similarity threshold to `_resolve_course_name()`
   - This fixes the critical bug causing incorrect results

### Medium Priority (Recommended):
2. **Fix #2**: Add configuration for threshold
   - Makes the system tunable without code changes

### Low Priority (Nice to Have):
3. Add input validation for tool parameters
4. Add logging for debugging course name resolution

---

## Test Coverage Assessment

The test suite successfully validates:
- ✅ Vector store search functionality
- ✅ Course metadata storage and retrieval
- ✅ Tool definition and execution
- ✅ AI generator tool calling flow
- ✅ End-to-end RAG system integration
- ✅ Session management
- ✅ Source tracking and attribution

The test suite identified:
- ❌ Course name resolution bug (semantic search always matches)
- ✅ All other components working as expected

---

## Conclusion

**The RAG chatbot is fundamentally working correctly**. The "query failed" issue reported by the user is likely due to:

1. **Primary cause**: Low Anthropic API credits (confirmed in server logs)
2. **Secondary issue**: Course name resolution bug (when filters are used with invalid course names)

**Fix #1 is essential** to prevent incorrect results when users specify invalid course names.

Once the API credit issue is resolved and Fix #1 is applied, the system should work correctly for content-related queries.

---

## How to Apply Fixes

1. **Stop the running server**: `Ctrl+C` on the running process
2. **Apply Fix #1**: Modify `backend/vector_store.py` as shown above
3. **Optionally apply Fix #2**: Add threshold to `config.py`
4. **Re-run tests**: `cd backend && uv run pytest tests/ -v`
5. **Restart server**: `./run.sh`
6. **Verify**: Test with queries like "What is in the XYZ Course?" (should say course not found)
