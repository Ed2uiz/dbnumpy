"""Stable, read-only backend capability descriptions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Execution features a frontend or planner may safely depend on."""

    backend_name: str
    sql_dialect: str
    semantic_nodes: frozenset[str]
    reductions: frozenset[str]
    native_pointwise_execution: bool
    arrow_ingestion: bool = True
    session_materialization: bool = True
    native_dimension_ranges: bool = True
    native_matrix_market_ingestion: bool = False
    external_parquet_scan: bool = False

    def supports_node(self, node: str) -> bool:
        return node in self.semantic_nodes

    def supports_reduction(self, operation: str) -> bool:
        return operation in self.reductions


V1_MATRIX_NODES = frozenset(
    {
        "source",
        "unary",
        "scalar_binary",
        "elementwise_binary",
        "transpose",
        "broadcast",
        "slice",
        "gather",
        "matmul",
    }
)

V1_REDUCTIONS = frozenset(
    {
        "sum",
        "mean",
        "var",
        "std",
        "min",
        "max",
        "any",
        "all",
        "nansum",
        "nanmean",
        "nanvar",
        "nanstd",
        "nanmin",
        "nanmax",
    }
)
