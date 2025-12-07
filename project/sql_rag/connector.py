from sqlalchemy import text
from typing import Dict, List, Any
from dbgpt_ext.datasource.rdbms.conn_sqlite import SQLiteConnector

class CustomSQLiteConnector(SQLiteConnector):
    def get_table_comment(self, table_name: str) -> Dict[str, str]:
        """Get table comments from db_comments table."""
        try:
            with self.session_scope() as session:
                cursor = session.execute(
                    text(
                        f"SELECT comment FROM db_comments WHERE object_type='TABLE' AND object_name='{table_name}'"
                    )
                )
                result = cursor.fetchone()
                if result:
                    return {"text": result[0]}
        except Exception as e:
            print(f"Error getting table comment for {table_name}: {e}")
        return {"text": ""}

    def get_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Get columns with comments from db_comments table."""
        columns = super().get_columns(table_name)
        try:
            with self.session_scope() as session:
                cursor = session.execute(
                    text(
                        f"SELECT object_name, comment FROM db_comments WHERE object_type='COLUMN' AND object_name LIKE '{table_name}.%'"
                    )
                )
                results = cursor.fetchall()
                comment_map = {}
                for row in results:
                    # object_name is table.column
                    col_name = row[0].split('.')[-1]
                    comment_map[col_name] = row[1]
                
                for col in columns:
                    if col['name'] in comment_map:
                        col['comment'] = comment_map[col['name']]
        except Exception as e:
            print(f"Error getting column comments for {table_name}: {e}")
        return columns
