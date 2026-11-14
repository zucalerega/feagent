"""The plan stage (dossier.loadcases): the load cases of the example, and each rule on a changed copy.

The tests of the planner's own rules use a stand-in for the hand-calculation engine (`StandIn`: `applies` reads
`when` as the contract says, `evaluate` gives a record without numbers), so that they test the planner and
not the formulas. `test_hand_calculations_of_the_example_with_the_real_engine` and the pipeline test run
`dossier.handcalc` itself.
"""

import re

import pytest
import yaml

from dossier import context, pipeline
from dossier.loadcases import plan, select
from dossier.model import MECHANICAL as _UNUSED  # noqa: F401  (keeps the import error clear if the model moves)
