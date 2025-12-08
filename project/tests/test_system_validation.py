#!/usr/bin/env python
"""
System validation script to check the reorganized structure and configuration.

This script validates that the system is properly set up without requiring
external services to be running.
"""

import sys
import os
from pathlib import Path
import json
import toml
from typing import Dict, Any, List

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))


def validate_folder_structure() -> Dict[str, bool]:
    """Validate the reorganized folder structure."""
    print("\n" + "=" * 60)
    print("VALIDATING FOLDER STRUCTURE")
    print("=" * 60)

    required_structure = {
        "retrieval/": "Main retrieval module",
        "retrieval/core/": "Core unified retrieval",
        "retrieval/sql_rag/": "SQL RAG module",
        "retrieval/doc_rag/": "Doc RAG module",
        "retrieval/context_provider/": "Context providers",
        "retrieval/context_provider/chart_schema/": "Chart schemas",
        "retrieval/context_provider/test_data_schema/": "Test data schemas",
        "output/": "Output directory",
        "output/logs/": "Log outputs",
        "output/charts/": "Chart outputs",
        "providers/": "Provider modules",
        "sql_rag/": "Original SQL RAG module",
        "doc_rag/": "Original Doc RAG module",
    }

    results = {}
    for path, description in required_structure.items():
        full_path = Path(path)
        exists = full_path.exists()
        results[path] = exists
        status = "✅" if exists else "❌"
        print(f"{status} {path:40} - {description}")

    return results


def validate_configuration() -> Dict[str, Any]:
    """Validate configuration files."""
    print("\n" + "=" * 60)
    print("VALIDATING CONFIGURATION")
    print("=" * 60)

    results = {}

    # Check config.toml
    config_path = Path("providers/config.toml")
    if config_path.exists():
        print("✅ config.toml found")
        try:
            with open(config_path, 'r') as f:
                config = toml.load(f)

            # Check key sections
            sections = ["providers", "qdrant", "testing"]
            for section in sections:
                if section in config:
                    print(f"  ✅ Section [{section}] present")
                    results[f"config_{section}"] = True
                else:
                    print(f"  ❌ Section [{section}] missing")
                    results[f"config_{section}"] = False

            # Check test database path
            if "testing" in config:
                test_db = config["testing"].get("test_db_url", "")
                # Remove environment variable syntax
                if "${env:" in test_db:
                    test_db = test_db.split(":-")[-1].rstrip("}")

                if test_db and Path(test_db).exists():
                    print(f"  ✅ Test database found: {test_db}")
                    results["test_database"] = True
                else:
                    print(f"  ⚠️  Test database not found: {test_db}")
                    results["test_database"] = False

        except Exception as e:
            print(f"  ❌ Error reading config.toml: {e}")
            results["config_valid"] = False
    else:
        print("❌ config.toml not found")
        results["config_exists"] = False

    # Check .env.dev
    env_path = Path("providers/.env.dev")
    if env_path.exists():
        print("✅ .env.dev found")
        results["env_exists"] = True

        # Check for API keys (without exposing them)
        with open(env_path, 'r') as f:
            env_content = f.read()
            keys = ["OPENAI_API_KEY", "AZURE_OPENAI_KEY"]
            for key in keys:
                if key in env_content:
                    print(f"  ✅ {key} configured")
                    results[f"env_{key}"] = True
                else:
                    print(f"  ❌ {key} missing")
                    results[f"env_{key}"] = False
    else:
        print("❌ .env.dev not found")
        results["env_exists"] = False

    return results


def validate_modules() -> Dict[str, bool]:
    """Validate Python modules can be imported."""
    print("\n" + "=" * 60)
    print("VALIDATING MODULE IMPORTS")
    print("=" * 60)

    modules_to_check = [
        ("retrieval.core.unified_retrieval", "UnifiedRetrievalSystem"),
        ("retrieval.sql_rag.sql_retrieval", "SQLRetrieval"),
        ("retrieval.doc_rag.doc_retrieval", "DocumentRetrieval"),
        ("providers.qdrant", "QdrantProvider"),
        ("providers.embeddings", "EmbeddingProvider"),
        ("providers.database", "DatabaseProvider"),
        ("providers.config", "load_config"),
        ("sql_rag.prompts.database_manager", "get_sql_generation_prompt"),
    ]

    results = {}
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
                results[module_name] = True
            else:
                print(f"❌ {module_name}.{class_name} - Class not found")
                results[module_name] = False
        except ImportError as e:
            print(f"❌ {module_name} - Import failed: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"❌ {module_name} - Error: {e}")
            results[module_name] = False

    return results


def validate_schema_files() -> Dict[str, Any]:
    """Validate schema and context files."""
    print("\n" + "=" * 60)
    print("VALIDATING SCHEMA FILES")
    print("=" * 60)

    results = {}

    # Check SQL schema
    sql_schema_path = Path("retrieval/context_provider/test_data_schema/output_schema.json")
    if sql_schema_path.exists():
        print(f"✅ SQL schema found: {sql_schema_path}")
        try:
            with open(sql_schema_path, 'r') as f:
                schema = json.load(f)

            if "tables" in schema:
                table_count = len(schema["tables"])
                print(f"  ✅ Schema contains {table_count} tables")

                # Check for importance_label (should be filtered)
                has_importance = False
                for table in schema["tables"]:
                    for column in table.get("columns", []):
                        if "importance_label" in column:
                            has_importance = True
                            break

                if has_importance:
                    print(f"  ℹ️  Schema contains importance_label (will be filtered)")
                else:
                    print(f"  ✅ No importance_label in schema")

                results["schema_valid"] = True
                results["table_count"] = table_count
                results["has_importance_label"] = has_importance
            else:
                print("  ❌ Schema missing 'tables' key")
                results["schema_valid"] = False

        except Exception as e:
            print(f"  ❌ Error reading schema: {e}")
            results["schema_valid"] = False
    else:
        print(f"❌ SQL schema not found: {sql_schema_path}")
        results["schema_exists"] = False

    # Check Vega schema
    vega_schema_path = Path("retrieval/context_provider/chart_schema/vega-lite-schema-v5.json")
    if vega_schema_path.exists():
        print(f"✅ Vega schema found: {vega_schema_path}")
        results["vega_schema_exists"] = True
    else:
        print(f"❌ Vega schema not found: {vega_schema_path}")
        results["vega_schema_exists"] = False

    return results


def check_services() -> Dict[str, bool]:
    """Check if required services are available."""
    print("\n" + "=" * 60)
    print("CHECKING EXTERNAL SERVICES")
    print("=" * 60)

    results = {}

    # Check Qdrant
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=2)
        if response.status_code == 200:
            print("✅ Qdrant is running on localhost:6333")
            results["qdrant_running"] = True
        else:
            print("⚠️  Qdrant responded but with status:", response.status_code)
            results["qdrant_running"] = False
    except Exception:
        print("❌ Qdrant is not running on localhost:6333")
        print("   Run: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        results["qdrant_running"] = False

    # Check for API key environment variables
    import os
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY environment variable set")
        results["openai_key_set"] = True
    else:
        print("⚠️  OPENAI_API_KEY not in environment (will use from .env.dev)")
        results["openai_key_set"] = False

    return results


def generate_sample_queries() -> List[str]:
    """Generate sample SQL queries for testing."""
    return [
        # Basic queries
        "Show me the top 5 customers by total revenue",
        "What was the total revenue for each month in 2018?",
        "List all product categories with their order counts",

        # Medium complexity
        "Find all sellers in São Paulo with their total sales",
        "Show products priced above their category average",
        "Calculate average delivery time by state",

        # Complex queries
        "Rank sellers within each state by revenue",
        "Calculate month-over-month growth for 2018",
        "Find top 10 product pairs bought together",
        "Segment customers into spending tiers with percentages",
        "Show year-over-year revenue growth by category"
    ]


def print_test_queries():
    """Print sample test queries."""
    print("\n" + "=" * 60)
    print("SAMPLE TEST QUERIES")
    print("=" * 60)

    queries = generate_sample_queries()
    for i, query in enumerate(queries, 1):
        print(f"{i:2}. {query}")


def main():
    """Main validation execution."""
    print("\n" + "=" * 80)
    print(" RETRIEVAL SYSTEM VALIDATION ".center(80))
    print("=" * 80)

    all_valid = True

    # Validate folder structure
    folder_results = validate_folder_structure()
    if not all(folder_results.values()):
        all_valid = False

    # Validate configuration
    config_results = validate_configuration()

    # Validate modules
    module_results = validate_modules()
    if not all(module_results.values()):
        all_valid = False

    # Validate schema files
    schema_results = validate_schema_files()

    # Check services
    service_results = check_services()

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    validation_counts = {
        "Folder Structure": sum(folder_results.values()),
        "Configuration": sum(v for k, v in config_results.items() if isinstance(v, bool) and v),
        "Module Imports": sum(module_results.values()),
        "Schema Files": sum(v for k, v in schema_results.items() if isinstance(v, bool) and v),
        "Services": sum(service_results.values())
    }

    for category, count in validation_counts.items():
        print(f"{category}: {count} checks passed")

    if all_valid and service_results.get("qdrant_running"):
        print("\n✅ System is ready for testing!")
        print("\nRun the test suite with:")
        print("  python test_retrieval_system.py")
    else:
        print("\n⚠️  Some validations failed or services are not running")

        if not service_results.get("qdrant_running"):
            print("\nTo start Qdrant:")
            print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")

        if not config_results.get("test_database"):
            print("\nDatabase not found. Update test_db_url in config.toml")

    # Print sample queries
    print_test_queries()

    print("\n" + "=" * 80)
    print(" END OF VALIDATION ".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()