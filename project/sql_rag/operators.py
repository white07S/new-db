
import json
from typing import Dict, Any, Tuple

from dbgpt.core.awel import MapOperator
from dbgpt.model.proxy import OpenAILLMClient
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.core.interface.llm import ModelRequest, ModelOutput

from . import logger
from .sql_validation import extract_allowed_tables, validate_sql_query

class RobustSQLOperator(MapOperator[Dict[str, Any], Dict[str, Any]]):
    """
    Executes SQL with retry and self-correction logic.
    """
    
    def __init__(self, connector, llm_client: OpenAILLMClient, model_name: str, max_retries: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.connector = connector
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_retries = max_retries

    async def map(self, input_value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input dict must contain:
        - sql: The SQL query to run
        - user_input: The user's original question
        - table_info: (Optional) schema info
        
        Returns:
        - data: execution result
        - sql: final sql (potentially corrected)
        - thoughts: reasoning
        """
        
        current_sql = input_value.get("sql", "")
        user_query = input_value.get("user_input", "")
        # table_info might be needed for correction context
        schema_context = input_value.get("table_info", "Schema info not available.")
        allowed_tables = extract_allowed_tables(schema_context)
        
        # We start with the thoughts from the previous step
        last_thoughts = input_value.get("thoughts", "")
        
        attempt = 0
        while attempt <= self.max_retries:
            is_valid, validation_error = validate_sql_query(current_sql, allowed_tables)
            if not is_valid:
                logger.warning(
                    "SQL validation failed",
                    extra={"props": {"reason": validation_error, "sql_preview": current_sql[:120]}},
                )
                attempt += 1
                if attempt > self.max_retries:
                    raise ValueError(f"SQL validation failed after retries: {validation_error}")
                current_sql, last_thoughts = await self._attempt_correction(
                    attempt, user_query, current_sql, validation_error, schema_context, last_thoughts
                )
                continue

            try:
                logger.info(
                    f"Attempt {attempt+1}: Executing SQL", extra={"props": {"sql": current_sql}}
                )
                df_result = self.connector.run_to_df(current_sql)
                logger.info("Execution Success")
                return {
                    **input_value,
                    "sql": current_sql,
                    "data": df_result,
                    "thoughts": last_thoughts,
                }
            except Exception as e:
                error_msg = str(e)
                logger.warning("Execution Failed", extra={"props": {"error": error_msg}})
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("Max retries reached. Raising exception")
                    raise
                current_sql, last_thoughts = await self._attempt_correction(
                    attempt, user_query, current_sql, error_msg, schema_context, last_thoughts
                )

        raise RuntimeError("SQL execution loop exited unexpectedly.")

    async def _attempt_correction(
        self,
        attempt_index: int,
        user_query: str,
        current_sql: str,
        error_msg: str,
        schema_context: str,
        last_thoughts: str,
    ) -> Tuple[str, str]:
        remaining_attempts = max(self.max_retries + 1 - attempt_index, 0)
        logger.info(
            "Triggering Correction",
            extra={"props": {"remaining_attempts": remaining_attempts}},
        )
        correction_result = await self._correct_sql(
            user_query,
            current_sql,
            error_msg,
            schema_context,
        )
        updated_sql = correction_result.get("sql", current_sql)
        new_thoughts = correction_result.get("thoughts", "")
        last_thoughts += (
            f"\n\n[Correction Attempt {attempt_index}]"
            f"\nError: {error_msg}\nFix Logic: {new_thoughts}"
        )
        return updated_sql, last_thoughts

    async def _correct_sql(self, query: str, bad_sql: str, error: str, schema: str) -> Dict[str, str]:
        """Call LLM to fix the SQL."""

        prompt = f"""You are a SQL Debugger.
User Question: {query}

Previous SQL:
{bad_sql}

Error Message:
{error}

Table Schema:
{schema}

Analyze the error and the previous SQL:
1. Identify the syntactical or logical error
2. Check the schema for correct table/column names
3. Write the CORRECTED SQL

Response Format (JSON):
{{
    "thoughts": "Explanation of why it failed and how to fix it...",
    "sql": "SELECT ..."
}}
"""

        messages = [ModelMessage(role=ModelMessageRoleType.HUMAN, content=prompt)]
        request = ModelRequest(
            model=self.model_name,
            messages=messages,
            temperature=0.0
        )

        response: ModelOutput = await self.llm_client.generate(request)
        if not response.success:
            raise Exception(f"Correction LLM call failed: {response.text}")

        text_resp = response.text

        # Parse JSON response
        try:
            start = text_resp.find("{")
            end = text_resp.rfind("}") + 1
            if start != -1 and end > start:
                json_str = text_resp[start:end]
                result = json.loads(json_str)
            else:
                result = json.loads(text_resp)

            logger.info(f"SQL Correction", extra={"props": {
                "thoughts": result.get("thoughts", ""),
                "corrected_sql": result.get("sql", "")[:100] + "..." if result.get("sql") else ""
            }})

            return result

        except Exception as e:
            logger.warning(f"Failed to parse correction JSON: {e}", extra={"props": {"response": text_resp[:500]}})
            # Fallback
            return {"sql": bad_sql, "thoughts": "Failed to parse correction."}
