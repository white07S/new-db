import json
from typing import Iterable, Optional, Set, Tuple

import sqlparse
from sqlparse import sql as sql_nodes
from sqlparse import tokens as sql_tokens

from dbgpt.util.sql_utils import remove_sql_comments

READONLY_KEYWORDS = {"SELECT", "WITH"}
BLOCKED_KEYWORDS = {
    "ALTER",
    "CREATE",
    "DELETE",
    "DROP",
    "INSERT",
    "MERGE",
    "REPLACE",
    "TRUNCATE",
    "UPDATE",
}
TABLE_TOKENS = {
    "FROM",
    "JOIN",
    "INNER JOIN",
    "LEFT JOIN",
    "RIGHT JOIN",
    "FULL JOIN",
    "CROSS JOIN",
    "UPDATE",
    "INTO",
}


def validate_sql_query(sql: Optional[str], allowed_tables: Optional[Set[str]] = None) -> Tuple[bool, str]:
    """Validate that the SQL is a single read-only statement referencing known tables."""

    cleaned = remove_sql_comments((sql or "")).strip()
    if not cleaned:
        return False, "Empty SQL body after removing comments."

    statements = sqlparse.parse(cleaned)
    if not statements:
        return False, "Failed to parse SQL statement."
    if len(statements) > 1:
        return False, "Multiple SQL statements detected; only one statement is allowed."

    statement = statements[0]
    first_token = statement.token_first(skip_cm=True)
    if not first_token:
        return False, "Could not identify the first token of the SQL statement."

    first_value = first_token.value.upper()
    leading_keyword = (
        first_value if first_token.ttype is sql_tokens.Keyword or first_token.ttype is sql_tokens.DML else first_value
    )

    if leading_keyword not in READONLY_KEYWORDS:
        return False, f"Only read-only SELECT/WITH statements are allowed (found '{leading_keyword}')."

    for token in statement.flatten():
        if token.ttype in (sql_tokens.Keyword, sql_tokens.Keyword.DDL, sql_tokens.Keyword.DML):
            value = token.value.upper()
            if value in BLOCKED_KEYWORDS:
                return False, f"Keyword '{value}' is not permitted in generated SQL."

    if allowed_tables:
        referenced_tables = _extract_referenced_tables(statement)
        unknown = [tbl for tbl in referenced_tables if tbl.lower() not in allowed_tables]
        if unknown:
            return False, f"SQL references unknown tables: {', '.join(sorted(set(unknown)))}."

    return True, ""


def _extract_referenced_tables(statement: sqlparse.sql.Statement) -> Iterable[str]:
    """Collect table names following FROM/JOIN style keywords."""

    tables = set()
    keyword_pending = None

    for token in statement.tokens:
        if token.is_whitespace or token.ttype in (sql_tokens.Newline, sql_tokens.Punctuation):
            continue

        if token.ttype is sql_tokens.Keyword:
            value = token.value.upper()
            if value in TABLE_TOKENS:
                keyword_pending = value
            else:
                keyword_pending = None
            continue

        if keyword_pending:
            tables.update(_identifier_names(token))
            keyword_pending = None
            continue

        if token.is_group:
            tables.update(_extract_referenced_tables(token))

    return tables


def _identifier_names(token) -> Set[str]:
    names: Set[str] = set()
    if isinstance(token, sql_nodes.IdentifierList):
        for identifier in token.get_identifiers():
            names.update(_identifier_names(identifier))
    elif isinstance(token, sql_nodes.Identifier):
        real = token.get_real_name() or token.get_name()
        if real:
            names.add(real)
    elif token.ttype is sql_tokens.Name:
        names.add(token.value)
    elif token.is_group:
        for child in token.tokens:
            names.update(_identifier_names(child))
    return names


def extract_allowed_tables(table_info: Optional[str]) -> Set[str]:
    """Parse the schema context JSON and return the table names."""
    allowed: Set[str] = set()
    if not table_info:
        return allowed
    parsed = None
    if isinstance(table_info, dict):
        parsed = table_info
    else:
        try:
            parsed = json.loads(table_info)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        allowed = {key.lower() for key in parsed.keys()}
    return allowed
