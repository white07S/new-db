"""
AWEL operators for chart generation with visualization support.
"""

import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from dbgpt.core.awel import MapOperator
from dbgpt.model.proxy import OpenAILLMClient

from retrieval.sql_rag.chart_agent import ChartGenerationAgent
from common.logger import setup_logger

logger = setup_logger("chart_operators")


class ChartGenerationOperator(MapOperator[Dict[str, Any], Dict[str, Any]]):
    """
    AWEL operator for chart generation with two-stage refinement.

    Input dict must contain:
    - data: DataFrame with query results
    - sql: Final SQL query
    - user_input: User's original question
    - thoughts: SQL agent's reasoning

    Returns:
    - Original input data plus:
    - chart_1: First chart visualization
    - chart_2: Second chart visualization
    - charts_saved_to: List of file paths where charts were saved
    """

    def __init__(
        self,
        llm_client: OpenAILLMClient,
        model_name: str,
        vega_schema_path: str = "chart_schema/vega-lite-schema-v5.json",
        output_dir: str = "chart_outputs",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_client = llm_client
        self.model_name = model_name
        self.vega_schema_path = vega_schema_path
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize chart generation agent
        self.chart_agent = ChartGenerationAgent(
            llm_client=llm_client,
            model_name=model_name,
            vega_schema_path=vega_schema_path
        )

    async def map(self, input_value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate charts from SQL query results.

        Args:
            input_value: Dictionary containing query results and context

        Returns:
            Dictionary with original data plus generated charts
        """
        try:
            # Extract required fields
            data = input_value.get("data")
            sql = input_value.get("sql", "")
            user_query = input_value.get("user_input", "")
            thoughts = input_value.get("thoughts", "")

            if data is None:
                logger.warning("No data provided for chart generation")
                return {
                    **input_value,
                    "chart_1": {},
                    "chart_2": {},
                    "charts_saved_to": [],
                    "chart_generation_error": "No data provided"
                }

            # Convert to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    logger.error(f"Unsupported data type: {type(data)}")
                    return {
                        **input_value,
                        "chart_1": {},
                        "chart_2": {},
                        "charts_saved_to": [],
                        "chart_generation_error": f"Unsupported data type: {type(data)}"
                    }

            logger.info(f"Generating charts for query: {user_query[:100]}...")
            logger.info(f"Data shape: {data.shape}")

            # Generate charts with two-stage refinement
            chart_results = await self.chart_agent.generate_charts(
                query=user_query,
                sql=sql,
                data=data,
                reasoning=thoughts,
                language="English",  # Can be made configurable
                custom_instruction=None
            )

            # Save charts to JSON files
            saved_files = await self._save_charts(
                chart_results,
                user_query,
                sql,
                data.shape
            )

            logger.info(f"Charts generated and saved to: {saved_files}")

            # Return results
            return {
                **input_value,
                "chart_1": chart_results.get("chart_1", {}),
                "chart_2": chart_results.get("chart_2", {}),
                "charts_metadata": chart_results.get("metadata", {}),
                "charts_saved_to": saved_files
            }

        except Exception as e:
            logger.error(f"Error in chart generation: {e}")
            return {
                **input_value,
                "chart_1": {},
                "chart_2": {},
                "charts_saved_to": [],
                "chart_generation_error": str(e)
            }

    async def _save_charts(
        self,
        chart_results: Dict[str, Any],
        query: str,
        sql: str,
        data_shape: tuple
    ) -> list:
        """
        Save generated charts to a single complete JSON file.

        Args:
            chart_results: Generated chart results
            query: User's query
            sql: SQL query
            data_shape: Shape of the dataframe (rows, columns)

        Returns:
            List of saved file paths (only complete file)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build complete output with all data
        complete_output = {
            "timestamp": timestamp,
            "query": query,
            "sql": sql,
            "data_shape": {
                "rows": data_shape[0],
                "columns": data_shape[1]
            },
            "metadata": chart_results.get("metadata", {}),
            "charts": []
        }

        # Process each chart - include full vega spec with data
        for chart_key in ["chart_1", "chart_2"]:
            if chart_key in chart_results and chart_results[chart_key]:
                chart = chart_results[chart_key]

                # Add complete chart with data to output
                chart_info = {
                    "chart_id": chart_key,
                    "reasoning": chart.get("reasoning", ""),
                    "chart_type": chart.get("chart_type", ""),
                    "vega_lite_spec": chart.get("chart_schema", {})  # This should include data
                }

                # Ensure the chart has data
                if chart_info["vega_lite_spec"] and "data" in chart_info["vega_lite_spec"]:
                    if "values" in chart_info["vega_lite_spec"]["data"]:
                        logger.info(f"{chart_key} has {len(chart_info['vega_lite_spec']['data']['values'])} data points")

                complete_output["charts"].append(chart_info)

        # Save only the complete results file with all data
        complete_file = os.path.join(
            self.output_dir,
            f"{timestamp}_chart_visualization.json"
        )

        with open(complete_file, "w") as f:
            json.dump(complete_output, f, indent=2)

        logger.info(f"Saved complete chart visualization to: {complete_file}")

        return [complete_file]


class ChartValidationOperator(MapOperator[Dict[str, Any], Dict[str, Any]]):
    """
    Operator for validating generated charts against Vega-Lite schema.
    This can be used as a separate validation step if needed.
    """

    def __init__(
        self,
        vega_schema_path: str = "chart_schema/vega-lite-schema-v5.json",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vega_schema_path = vega_schema_path

        # Load Vega-Lite schema
        with open(vega_schema_path, "r") as f:
            self.vega_schema = json.load(f)

    async def map(self, input_value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate charts in the input.

        Args:
            input_value: Dictionary potentially containing chart_1 and chart_2

        Returns:
            Dictionary with validation results added
        """
        from jsonschema import validate, ValidationError

        validation_results = {}

        for chart_key in ["chart_1", "chart_2"]:
            if chart_key in input_value:
                chart = input_value[chart_key]
                if chart and chart.get("chart_schema"):
                    try:
                        # Add required Vega-Lite schema reference
                        chart_schema = chart["chart_schema"].copy()
                        chart_schema["$schema"] = "https://vega.github.io/schema/vega-lite/v5.json"

                        # Validate against schema
                        validate(chart_schema, schema=self.vega_schema)
                        validation_results[f"{chart_key}_valid"] = True
                        validation_results[f"{chart_key}_validation_error"] = None
                        logger.info(f"{chart_key} validated successfully")
                    except ValidationError as e:
                        validation_results[f"{chart_key}_valid"] = False
                        validation_results[f"{chart_key}_validation_error"] = str(e)
                        logger.warning(f"{chart_key} validation failed: {e}")
                else:
                    validation_results[f"{chart_key}_valid"] = False
                    validation_results[f"{chart_key}_validation_error"] = "No chart schema provided"

        return {
            **input_value,
            "validation_results": validation_results
        }