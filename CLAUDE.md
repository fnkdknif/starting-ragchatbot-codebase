# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

A Course Materials RAG (Retrieval-Augmented Generation) system that enables semantic search and Q&A over educational content. The application uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Technology Stack

- **Backend**: FastAPI with uvicorn
- **AI/LLM**: Anthropic Claude (claude-sonnet-4-20250514)
- **Vector Store**: ChromaDB with persistent storage
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Frontend**: Vanilla JavaScript with static HTML/CSS
- **Package Manager**: uv (modern Python package manager)
- **Python Version**: 3.13+

## Essential Commands

### Development

```bash
# Install dependencies (first time setup)
uv sync

# Run the application (recommended)
./run.sh

# Or use make commands
make dev         # Run with hot reload
make run         # Run production server
```

The application serves both the API and frontend at `http://localhost:8000`:
- Web interface: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`

### Code Quality & Testing

```bash
# Run all quality checks (format + lint + type check)
make check

# Individual tools
make format      # Auto-format with black and ruff
make lint        # Run ruff linter (no auto-fix)
make mypy        # Run type checking

# Testing
make test                # Run all tests
make test-coverage       # Run tests with coverage report

# Utilities
make clean       # Remove cache and temporary files
make help        # Show all available commands
```

### Pre-commit Hooks (Optional)

To automatically run quality checks before each commit:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files

# Skip hooks for a single commit (use sparingly)
git commit --no-verify
```

### Environment Setup

Required environment variable in `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```

## Architecture Overview

### Component Design

The system follows a **modular component architecture** where the `RAGSystem` class (backend/rag_system.py) orchestrates interactions between specialized components:

```
RAGSystem (orchestrator)
├── DocumentProcessor - Parses course docs, extracts metadata, chunks content
├── VectorStore - ChromaDB wrapper for semantic search
├── AIGenerator - Anthropic Claude API wrapper with tool calling
├── SessionManager - Conversation history tracking
└── ToolManager - Manages and executes AI tools
    └── CourseSearchTool - Semantic search tool for Claude
```

### Data Model

The system uses a structured course hierarchy (backend/models.py):

- **Course**: Top-level container (title, instructor, course_link, lessons list)
- **Lesson**: Individual lesson (lesson_number, title, lesson_link)
- **CourseChunk**: Vector-stored content chunks (content, course_title, lesson_number, chunk_index)

### RAG Pipeline Flow

**Document Ingestion:**
1. Document processor parses course files (supports .txt, .pdf, .docx)
2. Extracts structured metadata (course title, instructor, lessons)
3. Content is split into overlapping chunks (800 chars, 100 char overlap)
4. Course metadata stored in `course_catalog` collection
5. Content chunks stored in `course_content` collection with embeddings

**Query Processing:**
1. User query received via `/api/query` endpoint
2. RAGSystem formats query for Claude with conversation history
3. Claude decides whether to use the `search_course_content` tool
4. If tool used: VectorStore performs semantic search with optional filters
5. Tool returns formatted results with course/lesson context
6. Claude synthesizes final response from search results
7. Session manager updates conversation history

### Two-Collection Vector Store Design

**Why Two Collections:**
- `course_catalog`: Stores course metadata (titles, instructors) for fuzzy course name matching
- `course_content`: Stores actual lesson content chunks for semantic search

This enables partial course name queries (e.g., "MCP" finds "Introduction to MCP") before searching content.

### Tool-Based RAG Approach

The system uses **Anthropic's tool calling** instead of traditional RAG prompt stuffing:

- Claude has access to a `search_course_content` tool with parameters: `query`, `course_name`, `lesson_number`
- Claude decides when to search based on the query type (general knowledge vs. course-specific)
- System prompt instructs Claude to use max one search per query
- Tool returns formatted results with source attribution
- Sources tracked separately for UI display via `ToolManager.get_last_sources()`

### Session Management

- Each user gets a session ID for conversation continuity
- Session stores last N exchanges (configurable via `MAX_HISTORY`)
- Conversation history passed to Claude as system prompt context
- Sessions stored in-memory (reset on server restart)

### Course Document Format

Expected format for files in `docs/` folder:

```
Course Title: [title]
Course Link: [url]
Course Instructor: [name]

Lesson 0: [lesson title]
Lesson Link: [url]
[lesson content...]

Lesson 1: [lesson title]
Lesson Link: [url]
[lesson content...]
```

The document processor intelligently extracts this structure. Files without lesson markers are treated as single documents.

## Configuration

All settings centralized in `backend/config.py`:

- `ANTHROPIC_MODEL`: Claude model version
- `EMBEDDING_MODEL`: Sentence transformer model
- `CHUNK_SIZE`: Content chunk size in characters (default: 800)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `MAX_RESULTS`: Number of search results returned (default: 5)
- `MAX_HISTORY`: Conversation exchanges to remember (default: 2)
- `CHROMA_PATH`: Vector database location (default: ./chroma_db)

## Code Quality Tools

The project uses industry-standard Python code quality tools:

### Black (Code Formatter)
- Automatic code formatting for consistent style
- Line length: 88 characters
- Configuration in `pyproject.toml` under `[tool.black]`
- Run: `make format` or `uv run black backend/`

### Ruff (Linter)
- Fast Python linter (replaces flake8, isort, pyupgrade)
- Checks for code style, potential bugs, and best practices
- Automatically fixes many issues with `--fix` flag
- Configuration in `pyproject.toml` under `[tool.ruff]`
- Run: `make lint` or `uv run ruff check backend/`

### Mypy (Type Checker)
- Static type checking for Python
- Helps catch type-related bugs before runtime
- Configured for gradual typing (not strict mode)
- Configuration in `pyproject.toml` under `[tool.mypy]`
- Run: `make mypy` or `uv run mypy backend/`

### Development Workflow

**Before committing code:**
```bash
make check  # Runs format + lint + mypy
```

**Or install pre-commit hooks:**
```bash
uv run pre-commit install
# Now quality checks run automatically on git commit
```

**Tool execution order:**
1. `black` - Formats code to consistent style
2. `ruff` - Lints and auto-fixes issues
3. `mypy` - Type checks the code

All tools are configured to work together harmoniously (e.g., ruff ignores E501 since black handles line length).

## Key Files

- `backend/app.py` - FastAPI application with endpoints and startup logic
- `backend/rag_system.py` - Main orchestrator coordinating all components
- `backend/vector_store.py` - ChromaDB interface with dual-collection design
- `backend/ai_generator.py` - Claude API wrapper with tool execution handling
- `backend/search_tools.py` - Tool definitions and execution (follows Anthropic tool spec)
- `backend/document_processor.py` - Course parsing and chunking logic
- `backend/session_manager.py` - Conversation history management
- `backend/models.py` - Pydantic data models
- `backend/config.py` - Configuration settings

## Important Implementation Details

### Startup Behavior

On application startup (`app.py:88-98`), the system automatically loads all documents from the `docs/` folder without clearing existing data. Duplicate courses are detected by title and skipped.

### Chunk Context Enhancement

Each chunk includes contextual prefix (document_processor.py:186, 234):
- First chunk of lesson: `"Lesson {N} content: {chunk}"`
- Subsequent chunks: `"Course {title} Lesson {N} content: {chunk}"`

This ensures chunks maintain context when retrieved independently.

### Search Resolution Logic

When searching with `course_name` parameter (vector_store.py:78-100):
1. Use semantic search on `course_catalog` to find best matching course title
2. Apply exact title filter to `course_content` search
3. Optionally filter by `lesson_number` if provided

This allows fuzzy matching ("MCP") while ensuring precise content filtering.

### Tool Execution Flow

AI generator handles multi-turn tool execution (ai_generator.py:89-135):
1. Initial Claude request with tool definitions
2. If `stop_reason == "tool_use"`, extract tool calls
3. Execute tools via ToolManager, collect results
4. Submit tool results back to Claude (new user message)
5. Claude generates final response without tools available

### Source Attribution

Sources flow from search tool to UI (search_tools.py:103-113):
- `CourseSearchTool` stores sources in `last_sources` attribute
- `ToolManager.get_last_sources()` retrieves them after query
- Returned to frontend via `/api/query` response
- `ToolManager.reset_sources()` clears for next query

## API Endpoints

- `POST /api/query` - Process user question, returns answer + sources + session_id
- `GET /api/courses` - Get course statistics (total count, titles list)

Request/response models defined in `app.py` using Pydantic.
