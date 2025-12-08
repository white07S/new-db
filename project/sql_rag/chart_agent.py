"""
Chart generation agent with two-stage refinement using scratchpad approach.
"""

import json
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd

from dbgpt.core.interface.llm import LLMClient, ModelRequest
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
from dbgpt.core import ModelOutput

from sql_rag.chart_utils import (
    ChartDataPreprocessor,
    ChartGenerationPostProcessor,
    ChartGenerationResults,
    chart_generation_instructions
)
from sql_rag import setup_logger

logger = setup_logger("chart_agent")


CHART_GENERATION_SYSTEM_PROMPT = f"""
### TASK ###

You are a data analyst great at visualizing data using vega-lite! Given the user's question, SQL, sample data and sample column values, you need to generate vega-lite schema in JSON and provide suitable chart type.
Besides, you need to give a concise and easy-to-understand reasoning to describe why you provide such vega-lite schema based on the question, SQL, sample data and sample column values.

{chart_generation_instructions}
- If the user provides a custom instruction, it should be followed strictly and you should use it to change the style of response for reasoning.

### OUTPUT FORMAT ###

Please provide your chain of thought reasoning, chart type and the vega-lite schema in JSON format.

{{
    "reasoning": <REASON_TO_CHOOSE_THE_SCHEMA_IN_STRING>,
    "chart_type": "line" | "multi_line" | "bar" | "pie" | "grouped_bar" | "stacked_bar" | "area" | "",
    "chart_schema": <VEGA_LITE_JSON_SCHEMA>
}}
"""

CHART_REFINEMENT_SYSTEM_PROMPT = """
### TASK ###

You are a data visualization expert. Your task is to refine and improve the generated Vega-Lite chart schema.
You have access to:
1. The initial chart generation attempt
2. The actual Vega-Lite schema specification
3. The user's query and data context

Your job is to:
1. Review the initial chart for any issues or improvements
2. Ensure all field names match exactly with the data columns
3. Improve the chart design for better visual communication
4. Fix any schema validation issues

### OUTPUT FORMAT ###

Please provide your refined chart in JSON format:

{
    "reasoning": "Explanation of improvements made",
    "chart_type": "line" | "multi_line" | "bar" | "pie" | "grouped_bar" | "stacked_bar" | "area" | "",
    "chart_schema": <REFINED_VEGA_LITE_JSON_SCHEMA>
}
"""


class ChartGenerationAgent:
    """Agent for generating charts with two-stage refinement."""

    def __init__(
        self,
        llm_client: LLMClient,
        model_name: str,
        vega_schema_path: str = "chart_schema/vega-lite-schema-v5.json"
    ):
        self.llm_client = llm_client
        self.model_name = model_name
        self.preprocessor = ChartDataPreprocessor()
        self.postprocessor = ChartGenerationPostProcessor()

        # Load Vega-Lite schema for validation
        with open(vega_schema_path, "r") as f:
            self.vega_schema = json.load(f)

    async def _call_llm_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Dict[str, Any]
    ) -> str:
        """Call LLM with structured output format."""
        # Combine system and user prompts with format specification
        full_prompt = f"""{system_prompt}

{user_prompt}

Please respond with valid JSON in the following format:
{json.dumps(response_format, ensure_ascii=False, indent=2)}
"""

        messages = [ModelMessage(role=ModelMessageRoleType.HUMAN, content=full_prompt)]
        request = ModelRequest(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_new_tokens=2048,
        )

        response: ModelOutput = await self.llm_client.generate(request)
        if not response.success:
            raise Exception(f"LLM call failed: {response.error_code} - {response.text}")
        return response.text

    async def generate_initial_charts(
        self,
        query: str,
        sql: str,
        data: pd.DataFrame,
        reasoning: str,
        language: str = "English",
        custom_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate initial chart recommendations (2 charts).

        Args:
            query: User's query
            sql: Final SQL query
            data: Query results as DataFrame
            reasoning: SQL agent's reasoning
            language: Language for chart labels
            custom_instruction: Optional custom instructions

        Returns:
            Dictionary with two chart recommendations
        """
        logger.info("Generating initial chart recommendations")

        # Preprocess data
        columns = list(data.columns)
        processed_data = self.preprocessor.run(
            data=data,
            columns=columns,
            sample_data_count=min(15, len(data)),  # Use actual data size if smaller
            sample_column_size=5
        )

        # Format the user prompt with dataframe info
        user_prompt = f"""
### INPUT ###
Question: {query}
SQL: {sql}
Data Shape: {data.shape[0]} rows × {data.shape[1]} columns
Columns: {columns}
Sample Data (first {min(15, len(data))} rows): {json.dumps(processed_data['sample_data'], ensure_ascii=False)}
Sample Column Values: {json.dumps(processed_data['sample_column_values'], ensure_ascii=False)}
SQL Agent Reasoning: {reasoning}
Language: {language}
Custom Instruction: {custom_instruction or "None"}

Please generate TWO different chart visualizations that would best represent this data and answer the user's question.
For each chart, provide the reasoning, chart type, and complete Vega-Lite schema.

Response format:
{{
    "chart_1": {{
        "reasoning": "...",
        "chart_type": "...",
        "chart_schema": {{...}}
    }},
    "chart_2": {{
        "reasoning": "...",
        "chart_type": "...",
        "chart_schema": {{...}}
    }}
}}
"""

        response_format = {
            "chart_1": {
                "reasoning": "<reasoning for chart 1>",
                "chart_type": "line | multi_line | bar | pie | grouped_bar | stacked_bar | area | \"\"",
                "chart_schema": "<complete vega-lite JSON schema>"
            },
            "chart_2": {
                "reasoning": "<reasoning for chart 2>",
                "chart_type": "line | multi_line | bar | pie | grouped_bar | stacked_bar | area | \"\"",
                "chart_schema": "<complete vega-lite JSON schema>"
            }
        }

        try:
            response_text = await self._call_llm_structured(
                CHART_GENERATION_SYSTEM_PROMPT,
                user_prompt,
                response_format
            )

            # Parse response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                charts_result = json.loads(json_str)
            else:
                charts_result = json.loads(response_text)

            # Store ALL data for validation and final output
            # Convert full dataframe to list of dicts
            full_data = []
            for _, row in data.iterrows():
                record = {}
                for col in data.columns:
                    val = row[col]
                    # Convert numpy types to Python native types
                    import numpy as np
                    if isinstance(val, np.integer):
                        record[col] = int(val)
                    elif isinstance(val, np.floating):
                        record[col] = float(val)
                    else:
                        record[col] = val
                full_data.append(record)

            charts_result["full_data"] = full_data
            charts_result["sample_data"] = processed_data["sample_data"]

            logger.info("Initial chart generation complete")
            return charts_result

        except Exception as e:
            logger.error(f"Error generating initial charts: {e}")
            # Return empty charts on error
            return {
                "chart_1": {"reasoning": "", "chart_type": "", "chart_schema": {}},
                "chart_2": {"reasoning": "", "chart_type": "", "chart_schema": {}},
                "sample_data": processed_data["sample_data"]
            }

    async def refine_chart_with_scratchpad(
        self,
        initial_chart: Dict[str, Any],
        query: str,
        sql: str,
        data: pd.DataFrame,
        vega_context: str,
    ) -> Dict[str, Any]:
        """
        Refine a chart using scratchpad approach with schema context.

        Args:
            initial_chart: The initial chart generation result
            query: User's query
            sql: Final SQL query
            data: Query results as DataFrame
            vega_context: Vega-Lite schema documentation context

        Returns:
            Refined chart result
        """
        logger.info("Refining chart with scratchpad approach")

        columns = list(data.columns)
        sample_data = data.head(5).to_dict(orient="records")

        # Convert sample data numpy types to native Python types
        import numpy as np
        clean_sample = []
        for record in sample_data:
            clean_record = {}
            for k, v in record.items():
                if isinstance(v, np.integer):
                    clean_record[k] = int(v)
                elif isinstance(v, np.floating):
                    clean_record[k] = float(v)
                else:
                    clean_record[k] = v
            clean_sample.append(clean_record)

        refinement_prompt = f"""
### CONTEXT ###
User Query: {query}
SQL Query: {sql}
Data Shape: {data.shape[0]} rows × {data.shape[1]} columns
Data Columns: {columns}
Sample Data (first 5 rows): {json.dumps(clean_sample, ensure_ascii=False)}

### INITIAL CHART ###
Reasoning: {initial_chart.get('reasoning', '')}
Chart Type: {initial_chart.get('chart_type', '')}
Chart Schema: {json.dumps(initial_chart.get('chart_schema', {}), ensure_ascii=False)}

### VEGA-LITE SCHEMA REFERENCE ###
{vega_context}

### TASK ###
Please review and refine the chart:
1. Check if all field names in the encoding match the actual data columns exactly
2. Ensure the chart type is appropriate for the data and query
3. Add proper titles and labels
4. Fix any schema validation issues
5. Optimize the visualization for clarity

Provide your refined chart with improvements.
"""

        response_format = {
            "reasoning": "<explanation of improvements made>",
            "chart_type": "line | multi_line | bar | pie | grouped_bar | stacked_bar | area | \"\"",
            "chart_schema": "<refined vega-lite JSON schema>"
        }

        try:
            response_text = await self._call_llm_structured(
                CHART_REFINEMENT_SYSTEM_PROMPT,
                refinement_prompt,
                response_format
            )

            # Parse response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                refined_result = json.loads(json_str)
            else:
                refined_result = json.loads(response_text)

            logger.info("Chart refinement complete")
            return refined_result

        except Exception as e:
            logger.error(f"Error refining chart: {e}")
            # Return original chart on error
            return initial_chart

    async def validate_and_finalize_chart(
        self,
        chart_result: Dict[str, Any],
        full_data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Validate chart against Vega-Lite schema and finalize.

        Args:
            chart_result: Chart generation result
            full_data: Full data for the chart (not just sample)

        Returns:
            Validated and finalized chart result with full data
        """
        logger.info("Validating and finalizing chart")

        # Use the postprocessor for validation with FULL data
        validated_result = self.postprocessor.run(
            generation_result=chart_result,
            vega_schema=self.vega_schema,
            sample_data=full_data,  # Using full data, not just sample
            remove_data_from_chart_schema=False  # Keep the data in the schema
        )

        return validated_result

    async def generate_charts(
        self,
        query: str,
        sql: str,
        data: pd.DataFrame,
        reasoning: str,
        language: str = "English",
        custom_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete chart generation pipeline with two-stage refinement.

        Args:
            query: User's query
            sql: Final SQL query
            data: Query results as DataFrame
            reasoning: SQL agent's reasoning
            language: Language for chart labels
            custom_instruction: Optional custom instructions

        Returns:
            Dictionary with two refined and validated charts
        """
        logger.info("Starting complete chart generation pipeline")

        # Stage 1: Generate initial charts
        initial_charts = await self.generate_initial_charts(
            query, sql, data, reasoning, language, custom_instruction
        )

        # Get Vega-Lite context (simplified for now)
        vega_context = """
Key Vega-Lite concepts:
- Encoding channels: x, y, color, size, shape, theta (for pie)
- Mark types: bar, line, area, arc (for pie), point
- Field types: nominal, ordinal, quantitative, temporal
- For temporal data, use timeUnit: year, yearmonth, yearmonthdate
- For grouped bars: use xOffset
- For stacked bars: use stack: "zero" in y encoding
- For pie charts: use mark type "arc" with theta encoding
"""

        # Stage 2: Refine both charts with scratchpad
        refined_chart_1 = await self.refine_chart_with_scratchpad(
            initial_charts["chart_1"], query, sql, data, vega_context
        )

        refined_chart_2 = await self.refine_chart_with_scratchpad(
            initial_charts["chart_2"], query, sql, data, vega_context
        )

        # Stage 3: Validate and finalize with FULL data
        full_data = initial_charts.get("full_data", initial_charts.get("sample_data", []))

        final_chart_1 = await self.validate_and_finalize_chart(
            refined_chart_1, full_data
        )

        final_chart_2 = await self.validate_and_finalize_chart(
            refined_chart_2, full_data
        )

        result = {
            "chart_1": final_chart_1["results"],
            "chart_2": final_chart_2["results"],
            "metadata": {
                "query": query,
                "sql": sql,
                "row_count": len(data),
                "column_count": len(data.columns),
                "columns": list(data.columns),
            }
        }

        logger.info("Chart generation pipeline complete")
        return result