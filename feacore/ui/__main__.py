"""``python -m feacore.ui --host 0.0.0.0 --port 8000`` - serve the API and the web UI."""

from __future__ import annotations

import argparse
import os


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m feacore.ui", description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose, e.g. on Brev)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--runs-dir", default=None, help="run store root (default $FEACORE_RUNS_DIR or <repo>/runs)")
    ap.add_argument("--quick-workers", type=int, default=None, help="parallel solve/geometry jobs (default 1)")
    ap.add_argument("--study-workers", type=int, default=None, help="parallel convergence studies (default 1)")
    ap.add_argument("--threads", type=int, default=None,
                    help="BLAS/OpenMP threads per worker process (sets OMP/MKL/OPENBLAS_NUM_THREADS)")
    ap.add_argument("--extra-problems", default=None,
                    help="comma-separated modules or .py files registering extra problems (scratch problems)")
    ap.add_argument("--token", default=None,
                    help="require this shared token on every /api/* request ('auto' generates one); "
                         "default: $FEACORE_UI_TOKEN, unset = no auth")
    ap.add_argument("--fetch-vendor", action="store_true",
                    help="download the pinned three.js files for offline use, then exit")
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args(argv)

    if args.fetch_vendor:
        from feacore.ui.server import fetch_vendor

        print(f"vendored three.js into {fetch_vendor()}")
        return
    if args.threads:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = str(args.threads)
    if args.extra_problems:
        os.environ["FEACORE_EXTRA_PROBLEMS"] = args.extra_problems
    if args.runs_dir:
        os.environ["FEACORE_RUNS_DIR"] = args.runs_dir  # child processes inherit it

    import uvicorn

    from feacore.ui.server import create_app

    token = args.token or os.environ.get("FEACORE_UI_TOKEN") or None
    if token == "auto":
        import secrets

        token = secrets.token_urlsafe(18)
    if token:
        os.environ["FEACORE_UI_TOKEN"] = token
    app = create_app(runs_dir=args.runs_dir, quick_workers=args.quick_workers, study_workers=args.study_workers,
                     token=token)
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print(f"feacore UI: {url}  (run store: {app.state.store.root})")
    if token:
        print(f"token auth ON. Share: {url}?token={token}  (or your tunnel URL + /?token=...)")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level,
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
