# Running feacore on NVIDIA Brev

Steps for a person or an agent on a Brev GPU instance. CLI commands are from the Brev docs
(https://docs.nvidia.com/brev/latest/cli/connectivity.md). The hackathon's own Brev notes
(`kit/use-brev/` in the NVIDIA ScienceClaw docs) take precedence where they differ.

## 1. Instance

Create one in the Brev console. Any single-GPU instance works; the GPU only speeds up large 3D
linear solves. CPU cores matter more, because studies run in parallel, one process per mesh.
Ubuntu 22.04 or 24.04, x86_64. Then, from your laptop:

```bash
brev shell <instance>                      # or: ssh <instance>
```

## 2. Code and setup (on the instance)

```bash
git clone https://github.com/zucalerega/feagent && cd feagent
git checkout fea-solver-benchmarks         # until it is merged
bash scripts/setup_gpu_box.sh              # ~3-5 min: apt libs + CalculiX, uv Python 3.12 venv, GPU extras
source .venv/bin/activate
```

The script installs `nvmath-python` (cuDSS) and CuPy matching the driver's CUDA version when
`nvidia-smi` works.

## 3. Check the GPU solve

```bash
python scripts/gpu_check.py
```

This solves the same systems with every available linear solver (SciPy, MKL PARDISO, cuDSS) and
requires agreement with SciPy to 1e-8, both on a 3D Laplacian and on a 3D benchmark solved end to
end. It must print `GPU CHECK PASSED` before cuDSS is used for anything that matters. To force a
solver: `export FEACORE_LINEAR_SOLVER=cudss` (or `scipy`, `pardiso`); the default `auto` prefers
cuDSS, then PARDISO, then SciPy.

## 4. Validate and export the benchmarks

```bash
python scripts/validate.py --jobs $(nproc)          # every problem, 3-mesh GCI study, report in docs/
python scripts/export_benchmarks.py                  # benchmarks/<id>/public + private
```

## 5. UI and solver service

```bash
export FEACORE_UI_TOKEN=$(openssl rand -hex 16)   # optional shared token; advised behind a tunnel
bash scripts/serve.sh                              # systemd service on port 8000, starts on boot
bash scripts/serve.sh logs                         # follow logs;  bash scripts/serve.sh stop
```

Expose port 8000 in one of two ways:
- **Share with the team (browser):** Brev console → your instance → Access → "Using Tunnels" →
  port 8000. You get a URL.
- **Just you:** on your laptop, `brev port-forward <instance> --port 8000:8000`, then open
  http://localhost:8000.

The mesh-harness / MCP side can call the solver over HTTP at the same address (see `docs/ui.md`).

## Team access (nothing to install on anyone's computer)

- **Browser (everyone):** create a tunnel for port 8000 (console → instance → Access → Using Tunnels)
  and share the URL, adding `?token=<FEACORE_UI_TOKEN>` if you set one. That gives the full UI
  (benchmarks, studies, batch runs, viewer, consensus). Scripts and agents on any computer can call
  the solver at the same URL: `POST <url>/api/solve` (case JSON + .msh); see the Connect panel and
  `docs/ui.md`. If the tunnel asks visitors to sign in, they need to be members of your Brev org.
- **Shell on the shared box:** invite teammates to your Brev org (console → Team → Invite Member).
  They install the Brev CLI once (`brew install brevdev/homebrew-brev/brev && brev login`), then
  run `brev shell <instance>`. Sharing rules for cloud instances can differ by org, so check the
  console's sharing options if a member can't see the instance.
- **Everyone gets their own identical box: a Launchable.** Console → Launchables → Create
  Launchable → VM mode, then paste `scripts/brev_launchable_setup.sh` as the setup script (it
  clones this public repo, runs the setup and starts the service on port 8000). Pick the GPU/CPU
  size (see below) and share the link. New Launchables default to "anyone with the link"; each
  deploy is a separate instance billed to whoever deploys it.

Sizing for 4 agents on 3D benchmarks such as NAFEMS LE10: 32 vCPUs, 128 GB RAM, one 48-80 GB
GPU, 200 GB disk (minimum about 16 vCPUs / 64 GB). The FEA is mostly CPU-bound; the GPU only
speeds up large 3D solves (cuDSS) and surrogate training.

## 6. Save credits

Stop the instance when you're not using it (Brev console, or `brev stop <instance>`; check
`brev --help` for the exact command in your CLI version). Commit and push results first: a stopped
or deleted instance may lose local files.

## DGX Spark (aarch64)

The same script works there and takes the conda-forge path for gmsh (PyPI has no Linux ARM wheel).
MKL PARDISO is skipped; cuDSS and SciPy remain.
