#!/usr/bin/env python3
"""Print a unique FEA problem statement for one agent.

    python make_problem.py <agent_number> <call_number>

The call number is hashed into a seed that shuffles every possible assignment; the agent number
indexes into that shuffle. Same inputs always give the same prompt, and different agents on the
same call never get the same assignment (up to the size of the assignment space).
"""

import hashlib
import itertools
import random
import sys

BCS = ["Fixed_Face_Base", "Fixed_Bolt_Holes", "Roller_Base_Z", "Pin_Bolt_Hole_A", "Pin_Bolt_Hole_B"]
FORCES = ["Axial_Load_X", "Bearing_Load_Hole_C", "Pressure_Face_Top", "Distributed_Torque_Y", "Point_Load_Edge_Tip"]
MESH_STRATEGIES = ["Linear Tets", "Quadratic Tets", "Quadratic Hexes", "Adaptive"]
SOLVERS = ["Direct", "Iterative PCG"]
SUBSET_SIZES = [1, 2]  # how many BCs / forces each agent is handed

PROMPT = """\
[SYSTEM]
You are an expert FEA configuration agent. Your objective is to setup a static structural analysis for the provided CAD part.
You must achieve the lowest possible computational cost (minimal mesh density, efficient solver settings) while strictly avoiding stress blowups, numerical divergence, or unconstrained rigid-body modes.

[ALLOWED CONSTRAINTS & FORCES]
- Available Boundary Conditions (BCs): {all_bcs}
- Available Force Types: {all_forces}

[ASSIGNED AGENT PROFILE & CONTEXT REROLL]
- Target Mesh Strategy: {mesh}
- Selected BC Tuple: {bcs}
- Selected Force Tuple: {forces}
- Target Solver: {solver}

[INSTRUCTIONS]
1. Apply the assigned BCs and forces to the CAD geometry.
2. If the combination is geometrically or physically invalid (e.g., causing a zero-pivot stiffness matrix, unconstrained free-body motion, or obvious geometric interferences), immediately output:
   `INVALID_CONFIG: [Reason for failure]`
3. If valid, estimate the computational cost (DOF count & estimated solve time) and predict whether stress concentrations will cause numerical blowup.
4. Output your configuration in structured JSON format for execution.
"""


def subsets(items):
    return [c for k in SUBSET_SIZES for c in itertools.combinations(items, k)]


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: make_problem.py <agent_number> <call_number>")
    agent, call = int(sys.argv[1]), int(sys.argv[2])

    space = list(itertools.product(MESH_STRATEGIES, SOLVERS, subsets(BCS), subsets(FORCES)))
    seed = int.from_bytes(hashlib.sha256(f"call:{call}".encode()).digest()[:8], "big")
    random.Random(seed).shuffle(space)
    mesh, solver, bcs, forces = space[agent % len(space)]

    print(PROMPT.format(
        all_bcs=", ".join(BCS),
        all_forces=", ".join(FORCES),
        mesh=mesh,
        bcs="(" + ", ".join(bcs) + ")",
        forces="(" + ", ".join(forces) + ")",
        solver=solver,
    ), end="")


if __name__ == "__main__":
    main()
