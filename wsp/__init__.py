"""wsp: teach a small open model to mesh, with a large teacher (Claude Opus) and the fixed solver.

Methods (see wsp/README.md):
  base        the open model as it is
  playbook    the teacher's meshing lessons in the prompt (ACE-style context, no weight change)
  wsp         weight-space programming: the teacher edits the weights directly, no gradients
              (Takehi, Teoh, Zhang, Khattab, "Sparks of Weight-Space Programming", MIT CSAIL 2026)
  sft         LoRA distillation of solver-verified teacher demonstrations
  rft         rejection-sampling fine-tuning: the student's own samples, kept when the solver scores them well
"""
