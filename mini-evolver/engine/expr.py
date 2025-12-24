"""Safe expression evaluation for constraint energies."""

from __future__ import annotations

import ast
import math
from typing import Any, Dict

ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "abs": abs,
}

ALLOWED_NAMES = {"pi": math.pi}


class ExprEvaluator(ast.NodeVisitor):
    def __init__(self, names: Dict[str, float]) -> None:
        self.names = names

    def visit(self, node: ast.AST) -> float:
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function")
        func = ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError("Unsupported function")
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in self.names:
            return float(self.names[node.id])
        if node.id in ALLOWED_NAMES:
            return float(ALLOWED_NAMES[node.id])
        raise ValueError(f"Unknown name: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Unsupported literal")

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError("Unsupported expression")


def eval_expr(expr: str, names: Dict[str, float]) -> float:
    tree = ast.parse(expr, mode="eval")
    evaluator = ExprEvaluator(names)
    return evaluator.visit(tree)
