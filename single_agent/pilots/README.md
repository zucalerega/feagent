# Pilot runs

Real-model analyses driven through `single_agent.cli` before the API loop has run. One JSON per arm holds:
- the agent's submitted report (idealization, answers, recommendations);
- every mesh request, every solve's setup and QoI values, and the studies;
- the scorecard from `single_agent.evaluate`.

Meshes and pictures are not kept.

| directory | problem | arms | driver |
|---|---|---|---|
| `abc_00000065_bc1/` | SC-40 / LC-06: the clamp body pressed onto two locating pins (holdout) | `opus.json`, `sonnet.json` | Claude Code subagents, handbook rules only, 2026-11-05/06 |

Results and the tool feedback that came out of them are in `../README.md` (Pilot).
