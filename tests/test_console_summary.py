import ast
import json
from pathlib import Path
from typing import Dict, List
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "eose_put_hedge_v3.py"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "console_summary_expected.json"


def _extract_projection_days(tree: ast.Module) -> int:
    """Find the FORWARD_PROJECTION_DAYS constant value from the module AST."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FORWARD_PROJECTION_DAYS":
                    if isinstance(node.value, ast.Constant):
                        return int(node.value.value)
    raise AssertionError("FORWARD_PROJECTION_DAYS constant not found")


def _load_build_console_summary():
    """Load build_console_summary without executing the full script."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    projection_days = _extract_projection_days(tree)

    func_def = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_console_summary"
    )

    minimal_module = ast.Module(body=[func_def], type_ignores=[])
    ast.fix_missing_locations(minimal_module)

    namespace = {
        "FORWARD_PROJECTION_DAYS": projection_days,
        "List": List,
        "Dict": Dict,
    }
    exec(compile(minimal_module, str(MODULE_PATH), "exec"), namespace)
    return namespace["build_console_summary"], projection_days


class BuildConsoleSummaryTest(unittest.TestCase):
    def test_console_summary_matches_checkpoint(self):
        build_console_summary, projection_days = _load_build_console_summary()
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

        recommendations = [
            {"dte": 30, "strike": 2.5, "contracts": 1, "premium": 0.35, "cost": 35},
            {"dte": 60, "strike": 2.0, "contracts": 2, "premium": 0.22, "cost": 44},
        ]

        summary = build_console_summary(
            ticker="EOSE",
            current_price=3.21,
            expected_low=2.10,
            expected_high=4.80,
            annual_vol=0.7420,
            recommendations=recommendations,
            hedge_benefit=5800,
            protection_pct=64.4,
        )

        self.assertEqual(projection_days, fixture["projection_days"])
        self.assertEqual(summary, fixture["summary"])


if __name__ == "__main__":
    unittest.main()
