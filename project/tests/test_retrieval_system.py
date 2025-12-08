"""
Test script for the unified retrieval system with complex SQL queries.

This script tests the complete retrieval system using configuration from
config.toml and .env.dev, demonstrating complex SQL generation capabilities.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import json
import pandas as pd
from dotenv import load_dotenv

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from retrieval.core.unified_retrieval import UnifiedRetrievalSystem, RetrievalConfig
from common.logger import setup_logger, get_logger

# Setup logging
logger = setup_logger(__name__)


class RetrievalSystemTester:
    """Test suite for the unified retrieval system."""

    def __init__(self):
        """Initialize the test suite."""
        # Load environment variables
        env_file = "providers/.env.dev"
        if Path(env_file).exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")

        # Initialize retrieval system with config
        self.system = UnifiedRetrievalSystem(
            config_path="providers/config.toml",
            env_file="providers/.env.dev"
        )

        # Test queries for SQL RAG
        self.sql_test_queries = [
            # Basic queries
            {
                "name": "Top Customers",
                "query": "Show me the top 5 customers by total revenue",
                "complexity": "basic"
            },
            {
                "name": "Monthly Revenue",
                "query": "What was the total revenue for each month in 2018?",
                "complexity": "basic"
            },

            # Multi-table JOIN queries
            {
                "name": "Customer Orders Join",
                "query": "Show me all customers from São Paulo who made orders in 2018 with their order totals",
                "complexity": "medium"
            },
            {
                "name": "Product Category Analysis",
                "query": """
                For each product category, show me:
                - Total revenue
                - Number of unique customers
                - Average delivery time in days
                - Average review score
                Only include delivered orders from 2018
                """,
                "complexity": "complex"
            },

            # Aggregation queries
            {
                "name": "Seller Performance",
                "query": """
                Calculate for each seller:
                - Total revenue
                - Number of orders
                - Number of unique customers
                - Average order value
                Sort by total revenue descending and show top 10
                """,
                "complexity": "medium"
            },
            {
                "name": "Category Performance with Filters",
                "query": """
                Find all product categories that have:
                - More than 100 orders
                - Average price above 50
                - Total revenue over 10000
                Show the category name, order count, average price, and total revenue
                """,
                "complexity": "complex"
            },

            # Window function queries
            {
                "name": "Seller Ranking by State",
                "query": """
                Rank sellers within each state by their total revenue.
                Show the top 3 sellers per state with their rank, state, city, and revenue.
                """,
                "complexity": "complex"
            },
            {
                "name": "Cumulative Revenue",
                "query": """
                Calculate the cumulative revenue by month for 2018.
                Include the percentage of total yearly revenue for each month.
                """,
                "complexity": "complex"
            },
            {
                "name": "Month-over-Month Growth",
                "query": """
                Compare each month's revenue with the previous month in 2018.
                Calculate the month-over-month growth rate percentage.
                """,
                "complexity": "complex"
            },

            # CTE queries
            {
                "name": "Customer Cohort Analysis",
                "query": """
                Find customer cohorts by their first purchase month.
                Show the number of customers in each cohort and their average lifetime value.
                """,
                "complexity": "complex"
            },
            {
                "name": "Seller Performance Comparison",
                "query": """
                Compare seller performance between first and second half of 2018.
                Include revenue growth, customer base growth, and average order value change.
                """,
                "complexity": "complex"
            },

            # Subquery queries
            {
                "name": "Above Average Products",
                "query": """
                Find products that are priced above the average price of their category.
                Show product name, category, price, and category average price.
                """,
                "complexity": "medium"
            },
            {
                "name": "Customer Order Analysis",
                "query": """
                For each customer who made more than 3 orders, show:
                - Customer ID
                - Total orders
                - Total spending
                - Their percentage of total revenue
                """,
                "complexity": "complex"
            },

            # Advanced analytics queries
            {
                "name": "Year-over-Year Growth",
                "query": """
                Calculate year-over-year revenue growth for each product category.
                Compare 2017 vs 2018 monthly revenue.
                Include growth percentage and absolute difference.
                """,
                "complexity": "complex"
            },
            {
                "name": "Customer Segmentation",
                "query": """
                Segment customers into tiers based on their total spending:
                - Platinum: top 10%
                - Gold: next 20%
                - Silver: next 30%
                - Bronze: remaining
                Show the count and average order value for each tier.
                """,
                "complexity": "complex"
            },
            {
                "name": "Market Basket Analysis",
                "query": """
                Find the top 10 product pairs that are frequently bought together.
                Show how many times each pair appears in the same order.
                Exclude pairs from the same category.
                """,
                "complexity": "complex"
            },
            {
                "name": "Geographic Sales Analysis",
                "query": """
                Analyze sales performance by state:
                - Total revenue
                - Number of orders
                - Average order value
                - Number of unique customers
                Group by state and sort by total revenue descending.
                """,
                "complexity": "medium"
            },

            # Complex business questions
            {
                "name": "Delivery Performance Analysis",
                "query": """
                Analyze delivery performance:
                - Average delivery time by state
                - Percentage of late deliveries (>10 days)
                - Correlation between order value and delivery time
                Group by seller state
                """,
                "complexity": "complex"
            },
            {
                "name": "Customer Retention Analysis",
                "query": """
                Calculate customer retention metrics:
                - Number of one-time customers
                - Number of repeat customers
                - Average time between first and second purchase
                - Repeat purchase rate by month
                """,
                "complexity": "complex"
            },
            {
                "name": "Product Performance Metrics",
                "query": """
                For the top 20 products by revenue, calculate:
                - Total units sold
                - Total revenue
                - Average price
                - Number of unique customers
                - Average review score
                - Return/cancellation rate
                """,
                "complexity": "complex"
            }
        ]

    async def test_sql_rag(self):
        """Test SQL RAG with complex queries."""
        logger.info("=" * 80)
        logger.info("TESTING SQL RAG SYSTEM")
        logger.info("=" * 80)

        # Initialize SQL RAG
        try:
            await self.system.initialize_sql_rag(recreate_collection=True)
            logger.info("✅ SQL RAG system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SQL RAG: {e}")
            return

        # Run test queries
        results = []
        for test_query in self.sql_test_queries:
            result = await self._run_sql_query(test_query)
            results.append(result)

        # Print summary
        self._print_sql_summary(results)

        return results

    async def _run_sql_query(self, test_query: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single SQL query test."""
        logger.info("-" * 60)
        logger.info(f"Test: {test_query['name']}")
        logger.info(f"Complexity: {test_query['complexity']}")
        logger.info(f"Query: {test_query['query'][:100]}..." if len(test_query['query']) > 100 else f"Query: {test_query['query']}")

        start_time = datetime.now()
        result = {
            "name": test_query["name"],
            "complexity": test_query["complexity"],
            "query": test_query["query"],
            "status": "pending"
        }

        try:
            # Execute query
            response = await self.system.query(test_query["query"], mode="sql")

            # Check success
            if response.get("success"):
                result["status"] = "success"
                result["rows"] = response.get("row_count", 0)
                result["columns"] = response.get("column_count", 0)
                result["attempts"] = response.get("attempts", 1)
                result["sql"] = response.get("sql", "")

                # Log results
                logger.info(f"✅ SUCCESS")
                logger.info(f"   Rows: {result['rows']}, Columns: {result['columns']}")
                logger.info(f"   SQL Attempts: {result['attempts']}")

                # Show sample of SQL
                sql_preview = result["sql"][:200] if len(result["sql"]) > 200 else result["sql"]
                sql_preview = sql_preview.replace('\n', ' ')
                logger.info(f"   SQL: {sql_preview}...")

                # If there's data, show sample
                if response.get("data") is not None and isinstance(response["data"], pd.DataFrame):
                    df = response["data"]
                    if not df.empty:
                        logger.info(f"   Sample Results (first 3 rows):")
                        sample = df.head(3).to_string()
                        for line in sample.split('\n'):
                            logger.info(f"      {line}")

            else:
                result["status"] = "failed"
                result["error"] = response.get("error", "Unknown error")
                logger.error(f"❌ FAILED: {result['error']}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"❌ ERROR: {e}")

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        result["execution_time"] = execution_time
        logger.info(f"   Execution Time: {execution_time:.2f}s")

        return result

    def _print_sql_summary(self, results: List[Dict[str, Any]]):
        """Print summary of SQL test results."""
        logger.info("=" * 80)
        logger.info("SQL RAG TEST SUMMARY")
        logger.info("=" * 80)

        # Count results by status
        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        error = sum(1 for r in results if r["status"] == "error")

        # Count by complexity
        complexity_counts = {}
        complexity_success = {}
        for r in results:
            complexity = r["complexity"]
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            if r["status"] == "success":
                complexity_success[complexity] = complexity_success.get(complexity, 0) + 1

        # Print statistics
        logger.info(f"Total Tests: {len(results)}")
        logger.info(f"✅ Successful: {success} ({success/len(results)*100:.1f}%)")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️  Errors: {error}")

        logger.info("\nBy Complexity:")
        for complexity in ["basic", "medium", "complex"]:
            if complexity in complexity_counts:
                total = complexity_counts[complexity]
                success = complexity_success.get(complexity, 0)
                logger.info(f"  {complexity.capitalize()}: {success}/{total} ({success/total*100:.1f}%)")

        # Show failed queries
        if failed > 0 or error > 0:
            logger.info("\nFailed/Error Queries:")
            for r in results:
                if r["status"] in ["failed", "error"]:
                    logger.info(f"  - {r['name']}: {r.get('error', 'Unknown error')[:100]}")

        # Performance statistics
        successful_results = [r for r in results if r["status"] == "success"]
        if successful_results:
            avg_time = sum(r["execution_time"] for r in successful_results) / len(successful_results)
            max_time = max(r["execution_time"] for r in successful_results)
            min_time = min(r["execution_time"] for r in successful_results)

            logger.info(f"\nPerformance:")
            logger.info(f"  Average Execution Time: {avg_time:.2f}s")
            logger.info(f"  Min Time: {min_time:.2f}s")
            logger.info(f"  Max Time: {max_time:.2f}s")

            # Average retry attempts
            avg_attempts = sum(r.get("attempts", 1) for r in successful_results) / len(successful_results)
            logger.info(f"  Average SQL Attempts: {avg_attempts:.2f}")

        logger.info("=" * 80)

    async def test_doc_rag(self):
        """Test Document RAG (if documents are available)."""
        logger.info("=" * 80)
        logger.info("TESTING DOCUMENT RAG SYSTEM")
        logger.info("=" * 80)

        # Initialize Doc RAG
        try:
            await self.system.initialize_doc_rag(recreate_collection=True)
            logger.info("✅ Document RAG system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Doc RAG: {e}")
            return

        # Look for test documents
        test_docs = []
        doc_paths = [
            "doc_rag/test_data/",
            "retrieval/context_provider/test_data/",
            "test_data/"
        ]

        for doc_path in doc_paths:
            path = Path(doc_path)
            if path.exists():
                # Find PDF, TXT, MD files
                for pattern in ["*.pdf", "*.txt", "*.md"]:
                    test_docs.extend(path.glob(pattern))

        if test_docs:
            logger.info(f"Found {len(test_docs)} test documents")

            # Index documents
            for doc in test_docs[:3]:  # Limit to first 3 for testing
                logger.info(f"Indexing: {doc.name}")
                result = await self.system.doc_retrieval.index_document(str(doc))
                if result.get("success"):
                    logger.info(f"  ✅ Indexed {result['chunks']} chunks")
                else:
                    logger.error(f"  ❌ Failed: {result.get('error')}")

            # Test queries
            test_queries = [
                "What are the main topics discussed in the documents?",
                "Summarize the key findings",
                "What recommendations are provided?"
            ]

            for query in test_queries:
                logger.info(f"\nQuery: {query}")
                result = await self.system.query(query, mode="doc")
                if result.get("success"):
                    logger.info(f"✅ Answer: {result['answer'][:200]}...")
                    logger.info(f"   Sources: {len(result.get('sources', []))} documents")
                else:
                    logger.error(f"❌ Failed: {result.get('error')}")

        else:
            logger.warning("No test documents found, skipping document RAG tests")

    def print_system_info(self):
        """Print system configuration and statistics."""
        logger.info("=" * 80)
        logger.info("SYSTEM INFORMATION")
        logger.info("=" * 80)

        stats = self.system.get_statistics()

        logger.info("Configuration:")
        for key, value in stats["configuration"].items():
            logger.info(f"  {key}: {value}")

        logger.info("\nCollections:")
        for collection, info in stats["collections"].items():
            logger.info(f"  {collection}: {info['document_count']} documents")

        logger.info("\nStatus:")
        logger.info(f"  SQL RAG: {'✅ Initialized' if stats['sql_rag']['initialized'] else '❌ Not initialized'}")
        logger.info(f"  Doc RAG: {'✅ Initialized' if stats['doc_rag']['initialized'] else '❌ Not initialized'}")

        if stats['sql_rag']['database']:
            logger.info(f"  Database: {stats['sql_rag']['database']}")

        logger.info("=" * 80)


async def main():
    """Main test execution."""
    logger.info("Starting Unified Retrieval System Tests")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")

    tester = RetrievalSystemTester()

    # Print system info
    tester.print_system_info()

    # Run SQL RAG tests
    sql_results = await tester.test_sql_rag()

    # Run Doc RAG tests (optional)
    # await tester.test_doc_rag()

    # Save results to file
    if sql_results:
        output_file = Path("output/logs") / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(sql_results, f, indent=2, default=str)

        logger.info(f"\nResults saved to: {output_file}")

    # Cleanup
    await tester.system.close()

    logger.info("\nTests completed!")


if __name__ == "__main__":
    asyncio.run(main())