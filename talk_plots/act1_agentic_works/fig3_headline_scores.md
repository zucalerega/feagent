# fig3_headline_scores: "The best agent scores 1.6-4.5x the automesher on every benchmark"

**Caption.** Base score of the automesher (0.25 by construction), the gen-0 mean agent, the last-generation mean agent and the best agent ever, per problem. Each problem uses the run that holds its best agent.

**Key numbers** (base score; the ratio is against the automesher's 0.25)
- Cook (deep1): 0.25 / 0.318 / 1.095 / **1.126 = 4.50x**.
- LE1 (split1): 0.25 / 0.782 / 0.487 / **1.062 = 4.25x**. The best is at gen 5, and the last generation is lower than gen 0.
- LE11 axisymmetric (deep1): 0.25 / 0.001 / 0.845 / **0.880 = 3.52x**. Gen 0 is the setup bug.
- LE11 3D (le11): 0.25 / 0.307 / 0.302 / **0.394 = 1.57x**. There is no training gain.

**What the score means** (ace/config.py)
- score = A / c, where c = DOFs / automesher DOFs.
- A = (g/2)^2 for g < 2, and A = g/2 above that.
- 1.0 = 2x the automesher's accuracy at its DOFs.
- Above the 2x target, a score ratio of k means "k x the accuracy per DOF". Cook's 4.5x is g = 4.76 at c = 2.11.

**Sources.** `act1_numbers.json` -> `numbers.problems.*.per_run`, `best_run`.

**Caveats**
- The best of all runs over all generations is a maximum over 44-60 runs per problem, so it is a selected value.
- For deep1 Cook the last-generation *mean* (1.095, 4.38x) is almost the same as the best, so selection is not what drives it.
- On LE1 and LE11 3D the best is an early, isolated peak.
- Identical meshes recur: LE1's best mesh also appears at g6-a0, and LE11-3D's at g7-a0 and g11-a0.
