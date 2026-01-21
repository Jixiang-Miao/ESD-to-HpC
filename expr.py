# expr.py
from typing import Union, List

class Expr:
    def eval(self, env: dict):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class Var(Expr):
    def __init__(self, name: str):
        self.name = name

    def eval(self, env):
        return env.get(self.name, None)

    def __str__(self):
        return self.name


class Const(Expr):
    def __init__(self, value: Union[int, float]):
        self.value = value

    def eval(self, env):
        return self.value

    def __str__(self):
        return str(self.value)


class BinOp(Expr):
    def __init__(self, left: Expr, op: str, right: Expr):
        self.left = left
        self.op = op
        self.right = right

    def eval(self, env):
        l = self.left.eval(env)
        r = self.right.eval(env)
        return eval(f"{l} {self.op} {r}")

    def __str__(self):
        return f"({self.left} {self.op} {self.right})"


class UnaryOp(Expr):
    def __init__(self, op: str, operand: Expr):
        self.op = op
        self.operand = operand

    def eval(self, env):
        val = self.operand.eval(env)
        if self.op == "¬":
            return not val
        return eval(f"{self.op}{val}")

    def __str__(self):
        return f"({self.op}{self.operand})"