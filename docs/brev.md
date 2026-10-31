# Running feagent on NVIDIA Brev (any account)

The stack here is the mesh harness, the fixed solver, verification, benchmarks, and the UI with its
solver service. It runs on any fresh Ubuntu box: your Brev instance, a teammate's Brev instance on
their own account, a Launchable, or a DGX Spark. It doesn't depend on whose account the box is in.

## Quick start (any Brev account, about 5 minutes)

1. **Create an instance** in your Brev console. Use VM mode, Ubuntu 22.04 or 24.04, and size it as
   below.
2. **Open a terminal on it:** the instance's Jupyter → File → New → Terminal, or `brev shell <name>`
   from a laptop with the Brev CLI.
3. **Run one command:**
   ```bash
   curl -fsSL https://raw.githubusercontent.com/zucalerega/feagent/main/scripts/bootstrap.sh | bash
   ```
   It clones the repo into `~/feagent`, installs everything (system libraries, CalculiX, a Python 3.12
   environment, GPU extras if a GPU is visible), and starts the UI/solver service on port 8000. The
   service restarts on failure, starts on boot, and updates itself from `main` within about a minute
   of each push.
4. **Open it:** Brev console → your instance → **HTTP Ports** → add port **8000**, then open the
   `https://…` link it shows. Brev asks visitors to sign in, so no extra token is needed. Don't use the
   "TCP/UDP → Open Port" dialog, which exposes the port to the whole internet without sign-in.

Running the same command again is safe: it pulls, re-installs anything missing, and restarts the
service.

## One-click for the whole team: a Launchable

In the Brev console: Launchables → Create Launchable → **VM mode**. Use this as the setup script:

```bash
#!/bin/bash
curl -fsSL https://raw.githubusercontent.com/zucalerega/feagent/main/scripts/bootstrap.sh | bash
```

Pick the size (below), add port 8000 if the wizard offers exposed ports, and share the link. Anyone
who opens it gets their own ready instance on their own Brev account, billed to them. New Launchables
default to "anyone with the link".

## Sizing

For 4 agents on 3D benchmarks such as NAFEMS LE10, use 32 vCPUs, 128 GB RAM, 200 GB disk and one GPU
(L40S / A100 / H100). The minimum is about 16 vCPUs / 64 GB. The FEA is mostly CPU-bound: the GPU only
speeds up large 3D solves (cuDSS) and surrogate training. If you run local LLMs on the same box, give
them the GPU and keep the FEA on the CPU with `export FEACORE_LINEAR_SOLVER=pardiso`.

## Day to day (on the box, in `~/feagent`)

```bash
bash scripts/serve.sh logs           # service log           (update-logs: the auto-update log)
bash scripts/serve.sh                # re-install / restart the service
bash scripts/serve.sh stop           # stop the service and the auto-update
.venv/bin/python scripts/gpu_check.py        # check the linear solvers (and the GPU) agree
.venv/bin/python scripts/validate.py --jobs $(nproc)   # reference GCI studies of every problem
```

- **Agents and scripts on the same box** call the solver at `http://localhost:8000` (`POST /api/solve`;
  the UI's Connect page has examples). That needs no sign-in or tunnel.
- **Teammates sharing one box** open its HTTP Ports link after you invite them to your Brev org
  (console → Team → Invite Member). From a laptop, scripts can use
  `brev port-forward <instance> --port 8000:8000` and then `http://localhost:8000`.
- **Token:** only needed if you expose the service without Brev's sign-in. Set
  `export FEACORE_UI_TOKEN=$(openssl rand -hex 16)` and re-run `bash scripts/serve.sh`; to remove it,
  `unset FEACORE_UI_TOKEN; rm -f .feacore_token; bash scripts/serve.sh`.
- **Auto-update** follows the branch checked out in `~/feagent` (`main` by default). It skips the
  update if you've edited files on the box, and it never restarts the service while jobs are running.

## GPU troubleshooting

`nvidia-smi` should list the GPU. If it says it "couldn't communicate with the NVIDIA driver":

- `lspci | grep -i nvidia` lists nothing: the instance has no GPU, and everything runs on the CPU.
- A GPU is listed: run `sudo reboot`, reconnect, then re-run the bootstrap command so it installs the
  GPU solver.

## DGX Spark (aarch64)

The same bootstrap works. PyPI has no Linux ARM wheel for gmsh, so it comes from conda-forge, and MKL
PARDISO is skipped; cuDSS and SciPy remain.

## Save credits

Stop the instance when you're not using it (Brev console, or `brev stop <instance>`). The service and
the auto-update come back by themselves when it starts. Push anything you want to keep first.
