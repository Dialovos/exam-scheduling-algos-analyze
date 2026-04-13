"""
Hard constraint violations (distance to feasibility) and soft constraint
penalty per ITC 2007 spec. Delegates to FastEvaluator for computation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from core.models import ProblemInstance, Solution


@dataclass
class EvaluationResult:
    """Detailed breakdown of solution quality."""
    # Hard constraint violations (Distance to Feasibility)
    conflicts: int = 0  # students with 2+ exams in same period
    room_occupancy: int = 0  # room capacity exceeded
    period_utilisation: int = 0  # exam duration > period duration
    period_related: int = 0  # AFTER/COINCIDENCE/EXCLUSION violations
    room_related: int = 0  # ROOM_EXCLUSIVE violations

    # Soft constraint penalties
    two_in_a_row: int = 0
    two_in_a_day: int = 0
    period_spread: int = 0
    non_mixed_durations: int = 0
    front_load: int = 0
    period_penalty: int = 0
    room_penalty: int = 0

    @property
    def hard_violations(self) -> int:
        return (self.conflicts + self.room_occupancy + self.period_utilisation +
                self.period_related + self.room_related)

    @property
    def is_feasible(self) -> bool:
        return self.hard_violations == 0

    @property
    def soft_penalty(self) -> int:
        return (self.two_in_a_row + self.two_in_a_day + self.period_spread +
                self.non_mixed_durations + self.front_load +
                self.period_penalty + self.room_penalty)

    @property
    def total_cost(self) -> int:
        """For comparison: feasibility first, then soft penalty."""
        return self.soft_penalty if self.is_feasible else float('inf')

    def summary(self) -> str:
        lines = [
            f"=== Evaluation Result ===",
            f"Feasible: {self.is_feasible}",
            f"Hard Violations: {self.hard_violations}",
            f"  Conflicts:          {self.conflicts}",
            f"  Room Occupancy:     {self.room_occupancy}",
            f"  Period Utilisation:  {self.period_utilisation}",
            f"  Period Related:     {self.period_related}",
            f"  Room Related:       {self.room_related}",
            f"Soft Penalty: {self.soft_penalty}",
            f"  Two in a Row:       {self.two_in_a_row}",
            f"  Two in a Day:       {self.two_in_a_day}",
            f"  Period Spread:      {self.period_spread}",
            f"  Non-Mixed Dur.:     {self.non_mixed_durations}",
            f"  Front Load:         {self.front_load}",
            f"  Period Penalty:     {self.period_penalty}",
            f"  Room Penalty:       {self.room_penalty}",
        ]
        return "\n".join(lines)


def evaluate(problem: ProblemInstance, solution: Solution) -> EvaluationResult:
    """Evaluate a solution against all hard and soft constraints.
    
    Now delegates to FastEvaluator for consistent results.
    """
    from core.fast_eval import FastEvaluator
    
    fe = FastEvaluator(problem)
    eb = fe.full_eval(solution)
    
    result = EvaluationResult()
    result.conflicts = eb.conflicts
    result.room_occupancy = eb.room_occupancy
    result.period_utilisation = eb.period_utilisation
    result.period_related = eb.period_related
    result.room_related = eb.room_related
    result.two_in_a_row = eb.two_in_a_row
    result.two_in_a_day = eb.two_in_a_day
    result.period_spread = eb.period_spread
    result.non_mixed_durations = eb.non_mixed_durations
    result.front_load = eb.front_load
    result.period_penalty = eb.period_penalty
    result.room_penalty = eb.room_penalty
    return result


def quick_feasibility_check(problem: ProblemInstance, solution: Solution) -> bool:
    """Fast check for hard constraint feasibility only."""
    result = evaluate(problem, solution)
    return result.is_feasible
