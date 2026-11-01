# fig1b_learning_curves_all_runs: "The deep loop learns steadily; the basic loop plateaus or drifts"

**Caption.** Every run scored on field error against the automesher (field mode), on every problem. Solid lines with dots are the best agent per generation. Dashed lines are the mean agent. Use this as a backup or honesty slide next to fig1.

**Key numbers** (gen 0 -> last generation; mean / best)
- Cook, deep1 (Opus teacher, 4 agents per task): 0.318 / 0.347 -> 1.095 / 1.126 by gen 10.
- Cook, cook_auto (basic loop, 6 agents): 0.449 / 0.449 -> 0.700 / 0.710 by gen 19. Its best ever is 0.771, at gen 11.
- Cook, split1 (basic loop, 3 agents per task): 0.392 / 0.453 -> 0.711 / 0.735 by gen 19. Its best ever is 0.801, at gen 2.
- LE1, split1: 0.782 / 0.888 -> 0.487 / 0.767 by gen 19. The best ever was 1.062, at gen 5. **There is no upward trend; the last generation is worse than gen 0.**
- LE11 axisymmetric, le11 (basic): 0.253 / 0.294 -> 0.091 / 0.272. The temperature-field setup bug keeps coming back (33 of its 57 runs have error > 20%).
- LE11 3D, le11 (basic): 0.307 / 0.342 -> 0.302 / 0.303. Flat; the best ever is 0.394.
- Cost: deep1 $46.37; cook_auto $2.09; split1 $3.98; le11 $5.21.

**Sources.** `ace_runs/{deep1,cook_auto,split1,le11}/gen*/runs/*.json` (checks). split1's `score` includes the bonus, so its base score = score - best_bonus (verified against the formula for all 120 runs).

**Caveats**
- The basic loops use a single-call Sonnet diagnoser, not the Opus teacher.
- The deep loop costs about 10-20x more per run.
- The runs differ in agents per task and in problems, so this is not a controlled ablation.
- The basic-loop runs got stuck on one recipe on Cook (base score near 0.71 for about 10 generations).
