import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "exp_value_u.py"


def evaluate_main_verdict(real_ci, sg_spread, eps):
    tree = ast.parse(SOURCE.read_text())
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ok_main" for target in node.targets)
    )
    expression = ast.Expression(assignment.value)
    return eval(
        compile(expression, str(SOURCE), "eval"),
        {"real_ci": real_ci, "SG_SPREAD": sg_spread, "EPS": eps},
    )


def test_spread_over_limit_fails_even_with_strong_ci():
    assert evaluate_main_verdict((0.9, 0.95), 0.4, 0.5) is False


def test_normal_spread_and_strong_ci_pass():
    assert evaluate_main_verdict((0.9, 0.95), 0.2, 0.5) is True


if __name__ == "__main__":
    test_spread_over_limit_fails_even_with_strong_ci()
    test_normal_spread_and_strong_ci_pass()
    print("PASS: both SG_SPREAD verdict scenarios")
