"""Queryable solve results: SavePlan, Trace, Result."""

from summer4.results.groups import SaveGroup, group_requests
from summer4.results.plan import (
    EVERYTHING,
    Compartments,
    ComputedValue,
    FlowMass,
    GroupedOutput,
    OutputShape,
    PlanDescription,
    SaveFn,
    SavePlan,
    SaveRequest,
)
from summer4.results.result import Result, ResultWithParams, SolverInfo
from summer4.results.targets import Target, TargetSet
from summer4.results.trace import Trace

__all__ = [
    "EVERYTHING",
    "Compartments",
    "ComputedValue",
    "FlowMass",
    "GroupedOutput",
    "OutputShape",
    "PlanDescription",
    "Result",
    "ResultWithParams",
    "SaveFn",
    "SaveGroup",
    "SavePlan",
    "SaveRequest",
    "SolverInfo",
    "Target",
    "TargetSet",
    "Trace",
    "group_requests",
]
