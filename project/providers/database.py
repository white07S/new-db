"""
Database Provider

This module provides a unified interface for database connections
and operations, currently supporting SQLite with potential for extension.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import sqlite3
import pandas as pd
from dbgpt_ext.datasource.rdbms.conn_sqlite import SQLiteConnector
from common.logger import get_logger

logger = get_logger(__name__)


class DatabaseProvider:
    """
    Unified database provider for SQL operations.
    Currently supports SQLite, can be extended for other databases.
    """

    def __init__(
        self,
        db_type: str = "sqlite",
        db_path: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        """
        Initialize the database provider.

        Args:
            db_type: Type of database ("sqlite", future: "postgres", "mysql")
            db_path: Path to the database file (for SQLite)
            connection_string: Connection string for other databases
        """
        self.db_type = db_type
        self.db_path = db_path
        self.connection_string = connection_string
        self.connector = None

        if db_type == "sqlite":
            if not db_path:
                raise ValueError("db_path is required for SQLite")
            self._init_sqlite()
        else:
            raise NotImplementedError(f"Database type {db_type} not yet supported")

        logger.info(f"Initialized database provider", extra={"props": {
            "db_type": db_type,
            "db_path": db_path
        }})

    def _init_sqlite(self):
        """Initialize SQLite connector."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"SQLite database not found: {self.db_path}")

        self.connector = SQLiteConnector.from_file_path(self.db_path)
        logger.info(f"Connected to SQLite database: {self.db_path}")

    def execute_query(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.

        Args:
            query: SQL query to execute
            params: Optional query parameters

        Returns:
            Query results as pandas DataFrame
        """
        try:
            if self.db_type == "sqlite":
                # DBGpt's SQLiteConnector handles parameterized queries
                df_result = self.connector.run_to_df(query, params)
                logger.debug(f"Query executed successfully, returned {len(df_result)} rows")
                return df_result
            else:
                raise NotImplementedError(f"Execute not implemented for {self.db_type}")

        except Exception as e:
            logger.error(f"Query execution failed: {e}", extra={"props": {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "error": str(e)
            }})
            raise

    def execute_scalar(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None
    ) -> Any:
        """
        Execute a SQL query and return a single value.

        Args:
            query: SQL query to execute
            params: Optional query parameters

        Returns:
            Single scalar value
        """
        try:
            df = self.execute_query(query, params)
            if not df.empty and len(df.columns) > 0:
                return df.iloc[0, 0]
            return None

        except Exception as e:
            logger.error(f"Scalar query failed: {e}")
            raise

    def execute_many(
        self,
        query: str,
        params_list: List[Union[tuple, dict]]
    ) -> int:
        """
        Execute a SQL query multiple times with different parameters.

        Args:
            query: SQL query to execute
            params_list: List of parameter sets

        Returns:
            Number of affected rows
        """
        try:
            affected_rows = 0
            for params in params_list:
                self.execute_query(query, params)
                affected_rows += 1

            logger.debug(f"Executed query {len(params_list)} times")
            return affected_rows

        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            raise

    def get_table_schema(
        self,
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get schema information for tables.

        Args:
            table_name: Specific table name or None for all tables

        Returns:
            Dictionary with schema information
        """
        try:
            if self.db_type == "sqlite":
                if table_name:
                    # Get columns for specific table
                    query = f"PRAGMA table_info({table_name})"
                    df = self.execute_query(query)

                    columns = []
                    for _, row in df.iterrows():
                        columns.append({
                            "name": row["name"],
                            "type": row["type"],
                            "nullable": row["notnull"] == 0,
                            "primary_key": row["pk"] == 1,
                            "default": row["dflt_value"]
                        })

                    return {
                        "table": table_name,
                        "columns": columns
                    }
                else:
                    # Get all tables
                    tables_query = """
                        SELECT name FROM sqlite_master
                        WHERE type='table'
                        AND name NOT LIKE 'sqlite_%'
                    """
                    tables_df = self.execute_query(tables_query)

                    schema = {}
                    for table in tables_df["name"]:
                        schema[table] = self.get_table_schema(table)["columns"]

                    return schema

            else:
                raise NotImplementedError(f"Schema retrieval not implemented for {self.db_type}")

        except Exception as e:
            logger.error(f"Failed to get table schema: {e}")
            raise

    def get_table_count(self, table_name: str) -> int:
        """
        Get the number of rows in a table.

        Args:
            table_name: Name of the table

        Returns:
            Number of rows
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            return self.execute_scalar(query)

        except Exception as e:
            logger.error(f"Failed to get table count: {e}")
            raise

    def get_sample_data(
        self,
        table_name: str,
        limit: int = 5
    ) -> pd.DataFrame:
        """
        Get sample data from a table.

        Args:
            table_name: Name of the table
            limit: Number of rows to retrieve

        Returns:
            Sample data as DataFrame
        """
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.execute_query(query)

        except Exception as e:
            logger.error(f"Failed to get sample data: {e}")
            raise

    def validate_query(self, query: str) -> bool:
        """
        Validate a SQL query without executing it.

        Args:
            query: SQL query to validate

        Returns:
            True if query is valid
        """
        try:
            if self.db_type == "sqlite":
                # Use EXPLAIN to validate without executing
                explain_query = f"EXPLAIN {query}"
                self.execute_query(explain_query)
                return True

        except Exception as e:
            logger.debug(f"Query validation failed: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if self.connector:
            if hasattr(self.connector, 'close'):
                self.connector.close()
                logger.info("Closed database connection")


# Convenience functions
def create_sqlite_provider(db_path: str) -> DatabaseProvider:
    """
    Create a SQLite database provider.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        DatabaseProvider instance
    """
    return DatabaseProvider(
        db_type="sqlite",
        db_path=db_path
    )


def create_database_provider(
    db_type: str,
    **kwargs
) -> DatabaseProvider:
    """
    Create a database provider based on type.

    Args:
        db_type: Type of database
        **kwargs: Additional parameters based on database type

    Returns:
        DatabaseProvider instance
    """
    return DatabaseProvider(
        db_type=db_type,
        **kwargs
    )