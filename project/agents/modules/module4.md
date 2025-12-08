Module 4: Extensible Response View System
Why this approach
A dict-based response system allows future extension without breaking changes. The base view handles text; extended views add structured data for charts, tables, and metadata.
Implementation
python# agents/response/views.py
from pydantic import BaseModel, Field
from typing import Any, Literal

class BaseView(BaseModel):
    """Minimum response structure"""
    text: str
    view_type: Literal["base"] = "base"

class DataView(BaseView):
    """Response with tabular data"""
    view_type: Literal["data"] = "data"
    data_dict: dict[str, list[Any]]          # Column-oriented data
    columns: list[str]
    row_count: int
    truncated: bool = False
    truncation_reason: str | None = None

class ChartView(DataView):
    """Response with visualization"""
    view_type: Literal["chart"] = "chart"
    chart_spec: dict                          # Vega-Lite specification
    chart_type: str                           # "bar", "line", "scatter", etc.

class ErrorView(BaseView):
    """Error response"""
    view_type: Literal["error"] = "error"
    error_code: str
    error_details: dict | None = None
    suggestions: list[str] = Field(default_factory=list)

class CompositeView(BaseModel):
    """Multiple views in one response"""
    views: list[BaseView | DataView | ChartView | ErrorView]
    primary_view_index: int = 0

# Type union for response handling
ResponseView = BaseView | DataView | ChartView | ErrorView | CompositeView

class ResponseBuilder:
    """Factory for creating response views"""
    
    @staticmethod
    def text(text: str) -> BaseView:
        return BaseView(text=text)
    
    @staticmethod
    def data(text: str, df_dict: dict, truncated: bool = False) -> DataView:
        return DataView(
            text=text,
            data_dict=df_dict,
            columns=list(df_dict.keys()),
            row_count=len(next(iter(df_dict.values()), [])),
            truncated=truncated
        )
    
    @staticmethod
    def chart(text: str, df_dict: dict, vega_spec: dict) -> ChartView:
        return ChartView(
            text=text,
            data_dict=df_dict,
            columns=list(df_dict.keys()),
            row_count=len(next(iter(df_dict.values()), [])),
            chart_spec=vega_spec,
            chart_type=vega_spec.get("mark", "unknown")
        )
