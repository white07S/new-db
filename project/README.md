# DBGpt RAG Testing System

A unified retrieval-augmented generation (RAG) system for SQL and document-based queries, built with modern vector database technology and advanced SQL generation capabilities.

## Features

- **Unified Retrieval System**: Single interface for both SQL and document RAG
- **Qdrant Vector Store**: High-performance vector search with hybrid capabilities
- **WrenAI-Style SQL Generation**: Advanced SQL query generation with complex operations support
- **Provider Pattern**: Centralized configuration for all external services
- **Schema Filtering**: Automatic removal of unnecessary metadata (importance_label)
- **Complex Query Support**: CTEs, window functions, joins, aggregations, and more

## Architecture

```
project/
├── providers/              # Centralized service providers
│   ├── config.toml        # Main configuration file
│   ├── .env.dev          # Environment variables
│   ├── config.py         # Configuration loader
│   ├── llm.py           # LLM provider (OpenAI/Azure)
│   ├── embeddings.py    # Embedding provider
│   ├── database.py      # Database provider
│   └── qdrant.py        # Qdrant vector store provider
│
├── retrieval/             # Core retrieval systems
│   ├── core/
│   │   └── unified_retrieval.py  # Unified interface
│   ├── sql_rag/
│   │   ├── sql_retrieval.py      # SQL RAG implementation
│   │   ├── prompts/              # SQL generation prompts
│   │   │   └── database_manager.py   # WrenAI-style prompts
│   │   ├── agent.py              # SQL agent
│   │   ├── operators.py         # SQL operators
│   │   ├── schema_search.py     # Schema search utilities
│   │   ├── sql_validation.py    # SQL validation
│   │   └── chart_*.py           # Chart generation utilities
│   └── doc_rag/
│       ├── doc_retrieval.py      # Document RAG implementation
│       ├── agent.py              # Document agent
│       ├── pdf_loader.py        # PDF document loader
│       ├── reranker.py          # Document reranking
│       └── test_data/           # Test documents
│
├── tests/                 # Test suite
│   ├── run_simple_test.py        # Simple SQL test runner
│   ├── test_retrieval_system.py  # Comprehensive test suite
│   ├── test_system_validation.py # System validation tests
│   ├── sql_rag/                  # SQL-specific tests
│   │   ├── schema_search_demo.py
│   │   ├── test_full_pipeline.py
│   │   └── test_sql_retry.py
│   └── doc_rag/                  # Document-specific tests
│       └── test_doc_rag_pipeline.py
│
├── common/                # Shared utilities
│   └── logger.py         # Logging configuration
│
├── agents/                # Agent modules
│   └── modules/          # Agent implementations
│
└── output/               # Output directories
    ├── logs/            # Application logs
    └── charts/          # Generated charts
```

## Prerequisites

- Python 3.8+
- Qdrant vector database running on localhost:6333
- OpenAI API key or Azure OpenAI configuration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Qdrant:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

3. Configure environment:
   - Copy `providers/.env.dev.example` to `providers/.env.dev`
   - Update with your API keys and database paths

## Configuration

The system uses `providers/config.toml` for configuration with environment variable support:

```toml
[providers]
default_provider = "openai"  # or "azure"

[providers.openai]
api_key = "${env:OPENAI_API_KEY}"

[qdrant]
host = "${env:QDRANT_HOST:-localhost}"
port = "${env:QDRANT_PORT:-6333}"
```

## Usage

### Quick Start

Run the simple test suite from the tests directory:
```bash
cd tests
python run_simple_test.py
```

### Comprehensive Testing

Run the full test suite with complex SQL queries:
```bash
cd tests
python test_retrieval_system.py
```

### Programmatic Usage

```python
from retrieval.core.unified_retrieval import UnifiedRetrievalSystem

# Initialize the system
system = UnifiedRetrievalSystem(
    config_path="providers/config.toml",
    env_file="providers/.env.dev"
)

# Initialize SQL RAG
await system.initialize_sql_rag(recreate_collection=True)

# Query the system
response = await system.query(
    "Show me the top 5 customers by total revenue",
    mode="sql"
)

if response["success"]:
    print(f"SQL: {response['sql']}")
    print(f"Results: {response['data']}")
```

## Supported SQL Operations

The system supports complex SQL queries including:

- **Basic Operations**: SELECT, WHERE, GROUP BY, ORDER BY
- **Joins**: INNER, LEFT, RIGHT, FULL OUTER
- **Aggregations**: SUM, AVG, COUNT, MIN, MAX
- **Window Functions**: ROW_NUMBER(), RANK(), DENSE_RANK(), LAG(), LEAD()
- **CTEs (Common Table Expressions)**: WITH clauses for complex queries
- **Subqueries**: Correlated and non-correlated subqueries
- **Advanced Analytics**: Year-over-year growth, cohort analysis, market basket analysis

## Test Queries Examples

The test suite includes 20+ complex query scenarios:

1. **Customer Analytics**
   - Top customers by revenue
   - Customer segmentation tiers
   - Retention analysis

2. **Sales Performance**
   - Monthly revenue trends
   - Year-over-year growth
   - Geographic analysis

3. **Product Analysis**
   - Category performance
   - Price comparisons
   - Market basket analysis

4. **Seller Metrics**
   - Performance rankings by state
   - Revenue comparisons
   - Customer base analysis

## Running Tests

All test files are located in the `tests/` directory:

```bash
# Navigate to tests directory
cd tests

# Run simple SQL test
python run_simple_test.py

# Run comprehensive test suite
python test_retrieval_system.py

# Run system validation (no external services required)
python test_system_validation.py

# Run SQL-specific tests
python sql_rag/test_full_pipeline.py
python sql_rag/test_sql_retry.py

# Run document RAG tests
python doc_rag/test_doc_rag_pipeline.py
```

## Development

### Adding New Providers

Create a new provider in `providers/` following the pattern:

```python
class NewProvider:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

    async def operation(self, **kwargs):
        # Implementation
```

### Extending SQL Generation

Modify prompts in `retrieval/sql_rag/prompts/database_manager.py`:
- Add new examples to `SQL_COMPLEX_EXAMPLES`
- Update system prompts in `SQL_GENERATION_SYSTEM_PROMPT`
- Add domain-specific instructions

### Custom Collections

The system supports multiple vector collections:
- `sql_schema`: Database schema embeddings
- `documents`: Document chunk embeddings
- Custom collections via `create_collection()`

## Troubleshooting

### Qdrant Connection Issues
```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Restart Qdrant
docker restart qdrant
```

### SQL Generation Errors
- Check logs in `output/logs/`
- Enable debug mode in config.toml
- Review generated SQL in test results

### Memory Issues
- Adjust batch size in configuration
- Use pagination for large result sets
- Implement streaming for document processing

## Performance

- **Vector Search**: ~50ms for 1000 documents
- **SQL Generation**: 1-3 seconds for complex queries
- **Hybrid Search**: Combines vector similarity and BM25 keyword search
- **Batch Processing**: Supports up to 100 documents per batch

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here]

## Acknowledgments

- WrenAI for SQL generation prompt inspiration
- DB-GPT for AWEL framework concepts
- Qdrant for vector database technology