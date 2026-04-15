"""
Knapsack Solver — Budget-Optimal Retention Action Selection
============================================================
Selects the best subset of retention interventions for a given
budget using binary (0/1) integer programming via PuLP, with a
greedy value-density fallback if PuLP is unavailable.

Usage:
    from src.optimization.knapsack_solver import KnapsackSolver

    solver = KnapsackSolver(budget_usd=300.0)
    result = solver.solve(items=[
        {"id": "price_cut",    "cost_usd": 45,  "value_usd": 480, "label": "Reduce price 15%"},
        {"id": "csm_assign",   "cost_usd": 120, "value_usd": 960, "label": "Assign CSM"},
        {"id": "loyalty_upg",  "cost_usd": 80,  "value_usd": 640, "label": "Loyalty upgrade"},
    ])
    # result["selected"] → list of chosen items
    # result["total_cost"] / result["total_value"]
"""

from __future__ import annotations

import structlog
from typing import Any

logger = structlog.get_logger(__name__)


class KnapsackSolver:
    """
    Binary knapsack solver for retention action portfolio optimisation.

    Args:
        budget_usd:    Maximum total spend allowed (the knapsack capacity).
        solver_name:   PuLP solver backend ('CBC' default, 'GLPK', etc.).
                       Ignored when falling back to the greedy algorithm.
    """

    def __init__(self, budget_usd: float = 300.0, solver_name: str = "CBC") -> None:
        self.budget_usd = budget_usd
        self.solver_name = solver_name

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def solve(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Select the value-maximising subset of *items* within *budget_usd*.

        Each item dict must contain:
            id        (str)   — unique identifier
            cost_usd  (float) — spend required to execute this action
            value_usd (float) — expected revenue saved / CLV preserved
            label     (str)   — human-readable description

        Returns a dict:
            selected       list[dict]  — chosen items (full item dicts)
            total_cost     float
            total_value    float
            roi            float       — (total_value - total_cost) / total_cost
            solver_used    str         — "pulp" | "greedy"
        """
        if not items:
            return self._empty_result("no_items")

        # Filter out items that individually exceed the budget
        feasible = [i for i in items if i["cost_usd"] <= self.budget_usd]
        if not feasible:
            return self._empty_result("no_feasible_items")

        try:
            return self._solve_pulp(feasible)
        except Exception as exc:
            logger.warning("PuLP solver failed — falling back to greedy", error=str(exc))
            return self._solve_greedy(feasible)

    # ------------------------------------------------------------------ #
    # PuLP integer-programming solver
    # ------------------------------------------------------------------ #

    def _solve_pulp(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        import pulp  # optional dependency — graceful fallback if absent

        n = len(items)
        prob = pulp.LpProblem("retention_knapsack", pulp.LpMaximize)

        # Binary decision variables: x_i ∈ {0, 1}
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        # Objective: maximise total value
        prob += pulp.lpSum(items[i]["value_usd"] * x[i] for i in range(n))

        # Budget constraint
        prob += pulp.lpSum(items[i]["cost_usd"] * x[i] for i in range(n)) <= self.budget_usd

        # Solve (suppress console output)
        solver = pulp.getSolver(self.solver_name, msg=False)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] not in ("Optimal", "Feasible"):
            raise RuntimeError(f"PuLP returned status: {pulp.LpStatus[prob.status]}")

        selected = [items[i] for i in range(n) if pulp.value(x[i]) > 0.5]
        return self._build_result(selected, solver_used="pulp")

    # ------------------------------------------------------------------ #
    # Greedy value-density fallback (value / cost, descending)
    # ------------------------------------------------------------------ #

    def _solve_greedy(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        # Sort by value density (value per dollar spent), descending
        ranked = sorted(items, key=lambda it: it["value_usd"] / max(it["cost_usd"], 1e-9), reverse=True)
        selected: list[dict[str, Any]] = []
        remaining = self.budget_usd

        for item in ranked:
            if item["cost_usd"] <= remaining:
                selected.append(item)
                remaining -= item["cost_usd"]

        return self._build_result(selected, solver_used="greedy")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_result(selected: list[dict[str, Any]], solver_used: str) -> dict[str, Any]:
        total_cost = sum(i["cost_usd"] for i in selected)
        total_value = sum(i["value_usd"] for i in selected)
        roi = round((total_value - total_cost) / max(total_cost, 1e-9), 4)
        return {
            "selected": selected,
            "total_cost": round(total_cost, 2),
            "total_value": round(total_value, 2),
            "roi": roi,
            "solver_used": solver_used,
        }

    @staticmethod
    def _empty_result(reason: str) -> dict[str, Any]:
        return {
            "selected": [],
            "total_cost": 0.0,
            "total_value": 0.0,
            "roi": 0.0,
            "solver_used": "none",
            "reason": reason,
        }
