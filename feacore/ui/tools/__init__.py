"""Dashboard tools: one module per dashboard section, each registering its whitelisted commands with
``feacore.ui.tasks.register_tool``. Modules here are imported by ``feacore.ui.tasks.discover_tools``;
names starting with ``_`` are skipped. See ``feacore/ui/tasks.py`` for the Tool / Param contract."""
