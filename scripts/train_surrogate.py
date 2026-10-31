#!/usr/bin/env python
"""Train the graph-network surrogate on a dataset from scripts/make_surrogate_dataset.py.

CPU proof of concept (this dev box, ~10 min):
  python scripts/train_surrogate.py --data /tmp/surrogate/ds1 --out models/surrogate/cpu \
      --device cpu --threads 2 --epochs 40 --max-minutes 12 --hidden 64 --layers 8

GPU (Brev), see docs/surrogate.md:
  python scripts/train_surrogate.py --data /data/surrogate/ds --out models/surrogate/gpu \
      --device cuda --epochs 300 --batch-size 16 --hidden 128 --layers 15 --lr 5e-4 --amp

Then evaluate (in- and out-of-distribution errors, timings, plots):
  python -m feacore.surrogate.evaluate --ckpt models/surrogate/cpu/best.pt --data /tmp/surrogate/ds1
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root (contains id/ and ood/)")
    ap.add_argument("--out", required=True, help="output dir for best.pt / last.pt / history.json / split.json")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=10, help="message-passing blocks")
    ap.add_argument("--no-global", action="store_true", help="disable the per-graph mean context")
    ap.add_argument("--model-type", default="mgn", choices=("mgn", "physicsnemo"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=None, help="cap the number of training graphs")
    ap.add_argument("--max-minutes", type=float, default=None, help="wall-time budget (LR schedule adapts)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (use 2 on the shared dev box)")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", metavar="CKPT", default=None, help="warm start from a checkpoint (same architecture)")
    ap.add_argument("--calibrate-only", metavar="CKPT", default=None,
                    help="only (re)fit the latent-space OOD model of an existing checkpoint")
    args = ap.parse_args(argv)

    import torch

    from feacore.surrogate.train import calibrate_latent, train

    if args.calibrate_only:
        lat = calibrate_latent(args.calibrate_only, args.data)
        print(json.dumps({"threshold": lat and lat["threshold"], "train_median": lat and lat["train_median"]}))
        return 0

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 2
    res = train(args.data, args.out, device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                lr_min=args.lr_min, weight_decay=args.weight_decay, hidden=args.hidden, layers=args.layers,
                global_context=not args.no_global, model_type=args.model_type, val_frac=args.val_frac,
                test_frac=args.test_frac, max_train=args.max_train, max_minutes=args.max_minutes, seed=args.seed,
                threads=args.threads, amp=args.amp, init=args.init)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
