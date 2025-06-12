#hpc.py
from typing import List, Union, Optional, Set

# ----- 前缀类型 -----


class Prefix:
    pass


class Wait(Prefix):
    def __init__(self, delay: float):
        self.delay = delay
    def __str__(self):
        return f"wait({self.delay})"

class Tau(Prefix):
    def __str__(self):
        return "τ"


class Channel:
    def __init__(self, name: str, para=None):
        self.name = name
        self.para = para
    def __str__(self):
        if self.para:
            return f"{self.name}({self.para})"
        else:
            return self.name

class InChannel(Channel):
    def __init__(self, name: str, para=None):
        super().__init__(name, para)
        self.io = "in"
    def __str__(self):
        if self.para:
            return f"{self.name}({', '.join(self.para)})"
        return f"{self.name}"

class OutChannel(Channel):
    def __init__(self, name: str, para=None):
        super().__init__(name, para)
        self.io = "out"
    def __str__(self):
        if self.para:
            return f"{self.name}⟨{', '.join(self.para)}⟩"
        return f"{self.name}\u0305"

class Input(Prefix):
    def __init__(self, channel: InChannel, var_list=None):
        self.channel = channel
        self.var_list = var_list

    def __str__(self):
        return f"{self.channel}({', '.join(self.var_list)})"


class Output(Prefix):
    def __init__(self, channel: OutChannel, expr_list=None):
        self.channel = channel
        self.expr_list = expr_list

    def __str__(self):
        return f"{self.channel}⟨{', '.join(self.expr_list)}⟩"


class Guard(Prefix):
    def __init__(self, condition: str):
        self.condition = condition

    def __str__(self):
        return f"[{self.condition}]"

class ODE:
    def __init__(self, e0: List[str], v: List[str], e: List[str], bound: str,
                 ready_set: Optional[List[Channel]] = None,
                 final_v: Optional[List[str]] = None):
        assert len(e0) == len(v) == len(e) == len(final_v or [])
        self.e0 = e0
        self.v = v
        self.e = e
        self.bound = bound
        self.ready_set = ready_set or []  # 避免 None
        self.final_v = final_v or []      # 避免 None

    def __str__(self):
        init_str = ", ".join(self.e0)
        deriv_str = ", ".join(f"{v}_dot={e}" for v, e in zip(self.v, self.e))
        ready_str = ", ".join(str(ch) for ch in self.ready_set)
        final_v_str = ", ".join(self.final_v)
        return f"{{{init_str} | {deriv_str} & {self.bound}}} ready={{{ready_str}}}.{final_v_str}"

class Continuous(Prefix):
    def __init__(self, ode: ODE):
        self.ode = ode

    def __str__(self):
        e0 = ", ".join(self.ode.e0)
        deriv_strs = [f"{var}_dot={deriv}" for var, deriv in zip(self.ode.v, self.ode.e)]
        deriv = ", ".join(deriv_strs)
        ready_channels = ", ".join(str(ch) for ch in self.ode.ready_set)
        return f"{{{e0} | {deriv} & {self.ode.bound}, {{{ready_channels}}}}}"
class Assignment(Prefix):
    def __init__(self, var: str, expr: str):
        self.var = var
        self.expr = expr

    def __str__(self):
        return f"⟨{self.var}:={self.expr}⟩"


# ----- 进程类型 -----
class Process:
    pass

class ProcessVariable(Process):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name


class NamedProcess(Process):
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def __str__(self):
        return f"{self.name} ::= {self.body}"

class Recursion(Process):
    def __init__(self, var, proc):
        self.var = var
        self.proc = proc

    def __str__(self):
        return f"{self.var}.{self.proc}"


class Inaction(Process):
    def __str__(self):
        return "0"


class PrefixProcess(Process):
    def __init__(self, prefix: Prefix, continuation: Process):
        self.prefix = prefix
        self.continuation = continuation

    def __str__(self):
        return f"{self.prefix}.{self.continuation}"


class Sum(Process):
    def __init__(self, branches: List[PrefixProcess]):
        self.branches = branches

    def __str__(self):
        return " + ".join(str(branch) for branch in self.branches)

class Parallel(Process):
    def __init__(self, processes: List[Process]):
        self.processes = processes

    def __str__(self):
        process_strs = []
        for p in self.processes:
            if isinstance(p, Process):
                process_strs.append(str(p))
            elif isinstance(p, (int, float)):
                continue  # Skip time values
        return " || ".join(process_strs)

class Restriction(Process):
    def __init__(self, names: List[str], process: Process):
        self.names = names
        self.process = process

    def __str__(self):
        return f"(ν {', '.join(self.names)}) {self.process}"


class Replication(Process):
    def __init__(self, process: Process):
        self.process = process

    def __str__(self):
        return f"!{self.process}"


class Conditional(Process):
    def __init__(self, condition: str, branch0: Process, branch1: Process):
        self.condition = condition
        self.branch0 = branch0
        self.branch1 = branch1
    def __str__(self):
        return f"if {self.condition} then {self.branch0} else {self.branch1}"

class Loop(Process):
    def __init__(self, channel: Channel, formal_paras: List[str], actual_paras: List[str], process: Process):
        self.channel = channel
        self.formal_paras = formal_paras
        self.actual_paras = actual_paras
        self.process = process
    def __str__(self):
        return f"μ{self.channel} {self.process}"


# 测试代码
if __name__ == "__main__":
    # Train ≜ (ν p, v, a)(overline{channels}⟨p,v,a⟩ . Run || Observer)

    train_output = PrefixProcess(
        Output(OutChannel("channels"), ["p", "v", "a"]),
        Parallel([Inaction(), Inaction()])

    )

    train = Restriction(["p", "v", "a"], train_output)
    print(train)
