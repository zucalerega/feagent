from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "benchmarks"
VERIFICATION_DIR = BENCH_DIR / "verification"
CALCULIX_DIR = BENCH_DIR / "calculix"
CODE_ASTER_DIR = BENCH_DIR / "code_aster"
SIMJEB_DIR = BENCH_DIR / "simjeb"
