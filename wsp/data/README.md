# wsp/data: the teacher's verified recipes

| file | what | written by |
|---|---|---|
| `teacher/<pid>.jsonl` | every recipe the teacher tried on a training part, in order, with its score (`x_auto`, gain, cost, error, DOFs) and a one-line rationale | the teacher (Claude Opus 5.5), via `python -m wsp.task score <pid> '<recipe>' --log ...` |
| `teacher/lessons_{A,B,C,D}.md` | each teacher's transferable lessons, backed by its measurements | the teacher |
| `demos.jsonl` | the valid recipes of every training part, re-scored and ranked (rank 0 = the part's best): the distillation targets | `python -m wsp.demos` |
| `playbook.md` | the lessons merged into one short list for the student's prompt (the in-context method) | the teacher, from the lessons files |

How the recipes were made: four teacher agents each took a group of part families (A: notched bars and
clevis lugs; B: stepped shafts; C: plates with holes; D: pressure vessels, Lamé cylinders and spheres,
cantilevers). For each training part a teacher read exactly the prompt the student gets
(`python -m wsp.task prompt <pid>`), wrote a recipe, scored it with the fixed solver against the fine
reference, and revised it, up to 8 scored attempts per part. The teachers saw only the prompts and their
own scores: no reference data, no test benchmark. The test benchmarks were never meshed by them.

A recipe's score is deterministic (same gmsh version, same solver), so `python -m wsp.demos` re-scores
every logged recipe rather than trusting the logged number.
