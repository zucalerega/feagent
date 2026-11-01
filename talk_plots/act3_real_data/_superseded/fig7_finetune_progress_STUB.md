# fig7_finetune_progress_STUB: "SLOT: fine-tuning the swarm on the real test (run pending)"

**Caption (placeholder).** Best and average agent error vs the real test per generation of fine-tuning, against
the automesher (16.6 %), the pre-fine-tuning bc0 mesh (4.5 %), the 2 % target and the 1.1 % model-error floor.

**Status.** The fine-tuning run has NOT happened. This file is a clearly marked slot, not a result.

**To fill it.** After the run:

    .venv/bin/python talk_plots/act3_real_data/make_figures.py        # if the inputs changed
    .venv/bin/python talk_plots/act3_real_data/plot_finetune_run.py ace_runs/<run> [--max-nodes N]

It writes fig7_finetune_progress, fig4b_error_vs_nodes_finetuned (fig4 + each generation's best mesh),
fig5b_curves_finetuned (fig5 + the best run's curve) and finetune_data.json here. Tested on a fake run dir
(mixed records, a failed run, a broken JSON, another problem's record, curves only in work/ or at a re-rooted
path, a missing error recomputed from the curve), then the fake dir was deleted.

**Caveats for when it is filled.** The best agent of a generation is a selection over N agents; also show the
average agent. Fine-tuning is scored against the same test that calibrated delta and c, so it is a fit to one
specimen, not a blind validation.
