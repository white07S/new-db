#!/usr/bin/env python
"""
Simple test runner to validate SQL query generation.
This demonstrates the system working with configuration from config.toml.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json
from dotenv import load_dotenv

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.core.unified_retrieval import UnifiedRetrievalSystem
from common.logger import setup_logger, get_logger

# Setup logging
logger = setup_logger(__name__)


async def run_simple_tests():
    """Run simple SQL tests to validate the system."""
    logger.info("=" * 80)
    logger.info("SIMPLE SQL QUERY TEST")
    logger.info("=" * 80)

    # Load environment variables
    env_file = "providers/.env.dev"
    if Path(env_file).exists():
        load_dotenv(env_file)
        logger.info(f"✅ Loaded environment from {env_file}")

    # Initialize system
    logger.info("\nInitializing Unified Retrieval System...")
    try:
        system = UnifiedRetrievalSystem(
            config_path="providers/config.toml",
            env_file="providers/.env.dev"
        )
        logger.info("✅ System initialized successfully")

        # Print configuration
        stats = system.get_statistics()
        logger.info(f"Provider: {stats['configuration']['provider']}")
        logger.info(f"LLM Model: {stats['configuration']['llm_model']}")
        logger.info(f"Embedding Model: {stats['configuration']['embedding_model']}")
        logger.info(f"Qdrant: {stats['configuration']['qdrant_host']}")

    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        return

    # Initialize SQL RAG
    logger.info("\nInitializing SQL RAG...")
    try:
        await system.initialize_sql_rag(recreate_collection=True)
        logger.info("✅ SQL RAG initialized and indexed")
    except Exception as e:
        logger.error(f"❌ Failed to initialize SQL RAG: {e}")
        logger.error(f"   Make sure Qdrant is running on localhost:6333")
        logger.error(f"   Run: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        return

    # Test queries
    test_queries = [
        "Show me the top 5 customers by total revenue",
        "What was the total revenue for each month in 2018?",
        "Find all products that are priced above the average price of their category",
        "Rank sellers within each state by their total revenue and show top 3 per state",
        "Calculate month-over-month revenue growth for 2018",
    ]

    logger.info("\n" + "=" * 80)
    logger.info("RUNNING TEST QUERIES")
    logger.info("=" * 80)

    results = []
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        logger.info("-" * 60)

        try:
            start_time = datetime.now()
            response = await system.query(query, mode="sql")
            execution_time = (datetime.now() - start_time).total_seconds()

            if response.get("success"):
                logger.info(f"✅ SUCCESS in {execution_time:.2f}s")

                # Show SQL snippet
                sql = response.get("sql", "")
                sql_lines = sql.split('\n')[:3]
                logger.info("Generated SQL (first 3 lines):")
                for line in sql_lines:
                    logger.info(f"  {line}")

                # Show result stats
                if response.get("data") is not None:
                    import pandas as pd
                    if isinstance(response["data"], pd.DataFrame):
                        df = response["data"]
                        logger.info(f"Results: {len(df)} rows, {len(df.columns)} columns")

                        # Show first few rows
                        if not df.empty and len(df) > 0:
                            logger.info("Sample data (first 3 rows):")
                            sample = df.head(3).to_string(max_cols=5)
                            for line in sample.split('\n'):
                                logger.info(f"  {line}")

                results.append({
                    "query": query,
                    "status": "success",
                    "time": execution_time,
                    "rows": len(df) if response.get("data") is not None else 0
                })

            else:
                error = response.get("error", "Unknown error")
                logger.error(f"❌ FAILED: {error}")
                results.append({
                    "query": query,
                    "status": "failed",
                    "error": error
                })

        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            results.append({
                "query": query,
                "status": "error",
                "error": str(e)
            })

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    success = sum(1 for r in results if r["status"] == "success")
    total = len(results)

    logger.info(f"Total Tests: {total}")
    logger.info(f"Successful: {success}/{total} ({success/total*100:.0f}%)")

    if success > 0:
        avg_time = sum(r.get("time", 0) for r in results if r["status"] == "success") / success
        logger.info(f"Average Query Time: {avg_time:.2f}s")

    # Save results
    output_dir = Path("output/logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"simple_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_file}")

    # Cleanup
    await system.close()
    logger.info("\nTest completed!")


if __name__ == "__main__":
    # Check if Qdrant is running
    import requests
    try:
        response = requests.get("http://localhost:6333/collections", timeout=2)
        if response.status_code == 200:
            print("✅ Qdrant is running")
            asyncio.run(run_simple_tests())
        else:
            print("⚠️ Qdrant responded but with unexpected status")
            print("Starting tests anyway...")
            asyncio.run(run_simple_tests())
    except Exception:
        print("❌ Qdrant is not running on localhost:6333")
        print("\nTo start Qdrant, run:")
        print("  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        print("\nOnce Qdrant is running, run this script again.")
        sys.exit(1)