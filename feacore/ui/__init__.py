"""Web UI + HTTP solver service for feacore. ``python -m feacore.ui --host 0.0.0.0 --port 8000``.

Modules: ``server`` (FastAPI app), ``jobs`` (child-process job runner), ``worker`` (job
bodies), ``viewdata`` (viewer triangulation / field encoding), ``analysis`` (verdicts,
submissions, consensus). The run store is ``feacore.runstore``.
"""


def create_app(*args, **kwargs):
    from feacore.ui.server import create_app as _create_app

    return _create_app(*args, **kwargs)
