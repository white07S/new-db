Module 5: Truncation Policy (DataFrame-specific)
Why this approach
Codex uses head+tail truncation for command output. For SQL results, we need schema-aware truncation that preserves high-priority columns and removes sparse data first.
Code Citation
File: codex-rs/ (truncation limits)
rustconst MODEL_FORMAT_MAX_LINES: usize = 256;
const MAX_BYTES: usize = 10 * 1024; // 10 KiB
Implementation
python# agents/truncation/dataframe_truncator.py
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

class ColumnPriority(BaseModel):
    """Column metadata from schema"""
    name: str
    priority: Literal["high", "medium", "low"] = "high"  # Default to high
    null_percentage: float = 0.0

class TruncationConfig(BaseModel):
    """Truncation policy configuration"""
    max_rows: int = 500
    max_columns: int = 50
    max_bytes: int = 50 * 1024                           # 50KB
    null_threshold: float = 0.8                          # 80% null removal
    preserve_all_high_priority: bool = True

class TruncationResult(BaseModel):
    """Result of truncation operation"""
    data: dict[str, list]
    original_rows: int
    original_columns: int
    final_rows: int
    final_columns: int
    removed_columns: list[str]
    truncated: bool
    reason: str | None = None

class DataFrameTruncator:
    """Three-step truncation policy for SQL results"""
    
    def __init__(self, schema: dict, config: TruncationConfig = None):
        self.schema = schema
        self.config = config or TruncationConfig()
        self._priority_map = self._build_priority_map()
    
    def _build_priority_map(self) -> dict[str, str]:
        """Extract column priorities from schema"""
        priority_map = {}
        for table in self.schema.get("tables", []):
            for col in table.get("columns", []):
                priority_map[col["name"]] = col.get("priority", "high")
        return priority_map
    
    def truncate(self, df: pd.DataFrame) -> TruncationResult:
        """Apply three-step truncation policy"""
        original_rows, original_cols = df.shape
        removed_columns = []
        reason = None
        
        # Step 1: Remove columns with >80% null/nan/empty
        null_percentages = df.isnull().sum() / len(df)
        empty_percentages = (df == "").sum() / len(df)
        sparse_cols = (null_percentages + empty_percentages) > self.config.null_threshold
        
        cols_to_remove_step1 = df.columns[sparse_cols].tolist()
        df = df.drop(columns=cols_to_remove_step1)
        removed_columns.extend(cols_to_remove_step1)
        
        if cols_to_remove_step1:
            reason = f"Removed {len(cols_to_remove_step1)} sparse columns (>80% null)"
        
        # Check if still too large
        if self._estimate_size(df) <= self.config.max_bytes:
            return self._build_result(df, original_rows, original_cols, removed_columns, reason)
        
        # Step 2: Remove columns by priority (low → medium, NEVER high)
        for priority in ["low", "medium"]:
            if self._estimate_size(df) <= self.config.max_bytes:
                break
            
            cols_to_remove = [
                col for col in df.columns
                if self._priority_map.get(col, "high") == priority
            ]
            df = df.drop(columns=cols_to_remove)
            removed_columns.extend(cols_to_remove)
            
            if cols_to_remove:
                reason = f"Removed {priority}-priority columns for size"
        
        # Check again
        if self._estimate_size(df) <= self.config.max_bytes:
            return self._build_result(df, original_rows, original_cols, removed_columns, reason)
        
        # Step 3: Truncate ROWS (preserve all remaining columns)
        if len(df) > self.config.max_rows:
            df = df.head(self.config.max_rows)
            reason = f"Truncated to {self.config.max_rows} rows (columns preserved)"
        
        return self._build_result(df, original_rows, original_cols, removed_columns, reason)
    
    def _estimate_size(self, df: pd.DataFrame) -> int:
        """Estimate serialized size in bytes"""
        return df.memory_usage(deep=True).sum()
    
    def _build_result(self, df, orig_rows, orig_cols, removed, reason) -> TruncationResult:
        return TruncationResult(
            data=df.to_dict(orient="list"),
            original_rows=orig_rows,
            original_columns=orig_cols,
            final_rows=len(df),
            final_columns=len(df.columns),
            removed_columns=removed,
            truncated=len(removed) > 0 or len(df) < orig_rows,
            reason=reason
        )
Schema Priority Example
json{
  "tables": [{
    "name": "orders",
    "columns": [
      {"name": "order_id", "type": "int", "priority": "high"},
      {"name": "customer_id", "type": "int", "priority": "high"},
      {"name": "total_amount", "type": "decimal", "priority": "high"},
      {"name": "created_at", "type": "timestamp", "priority": "medium"},
      {"name": "internal_notes", "type": "text", "priority": "low"},
      {"name": "debug_flags", "type": "json", "priority": "low"}
    ]
  }]
}