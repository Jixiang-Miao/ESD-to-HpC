#hpc.py
from typing import List, Union, Optional, Set
from expr import Expr, UnaryOp, Var, BinOp

class Prefix:
    pass

class Wait(Prefix):
    def __init__(self, delay: float):
        self.delay = delay
        
    def to_restriction(self, t_var="t"):
        # Convert wait(d) to (νt){0|ṫ=1&t<d}
        # t_var = "t"  # Fresh variable name for the timer
        ode = ODE(
            e0=["0"],  # Initial value
            v=[t_var],  # Variable
            e=["1"],    # Derivative (ṫ=1)
            bound=f"{t_var}<{self.delay}",  # Boundary condition
            final_v=[t_var]  # Final value
        )
        return Restriction(
            names=[t_var],
            process=Continuous(ode))
            
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
        if self.name.startswith("load("):
            var_name = self.name[5:-1]
            return f"{var_name}"
        return super().__str__()

class OutChannel(Channel):
    def __init__(self, name: str, para=None):
        super().__init__(name, para)
        self.io = "out"

    def __str__(self):
        if self.name.startswith("allocation("):
            var_name = self.name[11:-1]
            return f"{var_name}\u0305"
        elif self.name.startswith("store("):
            var_name = self.name[6:-1]
            return f"{var_name}"
        return super().__str__()
    

class Input(Prefix):
    def __init__(self, channel: InChannel, var_list=None):
        self.channel = channel
        self.var_list = var_list or []
    def __str__(self):
        return f"{self.channel}({', '.join(self.var_list)})"


class Output(Prefix):
    def __init__(self, channel: OutChannel, expr_list=None):
        self.channel = channel
        self.expr_list = expr_list or []

    def __str__(self):
        expr_strs = [str(expr) for expr in self.expr_list]
        return f"{self.channel}⟨{', '.join(expr_strs)}⟩"


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
        self.ready_set = ready_set or []
        self.final_v = final_v or []

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
    def __init__(self, var: Var, expr: Expr):
        self.var = var
        self.expr = expr
        
    def to_restriction(self, is_first: bool = True):
        terminus = f"{self.var.name}"
        
        if is_first:
            # (ν terminus) allocation(var)⟨expr⟩.terminus(x).store(var)⟨x⟩
            allocation = Output(OutChannel(f"allocation({self.var.name})"), [self.expr])
            store = Output(OutChannel(f"store({self.var.name})"), [Var("x")])
            
            process = allocation
        else:
            # (ν terminus) load(var)(x).terminus⟨x+expr⟩.store(var)⟨x+expr⟩
            load = Output(OutChannel(f"{self.var.name}"), [self.expr])
            sum_expr = BinOp(Var("x"), "+", self.expr)
            output_terminus = Output(OutChannel(terminus), [sum_expr])
            store = Output(OutChannel(f"store({self.var.name})"), [self.expr])
            
            process = load
            
        return Restriction(
            names=[terminus],
            process=process
        )
        
    def __str__(self):
        return f"⟨{self.var}:={self.expr}⟩"
    
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
    def __init__(self, var: str, params: List[str], proc: Process, actual_params: List[str] = None):
        self.var = var
        self.params = params or []
        self.proc = proc
        self.actual_params = actual_params or []
        
    def to_restriction(self):
        """Convert μx(y).P@<e> to (νx)(x̅<e> || !x(y).P)"""
        # Create the output prefix (x̅<e>)
        output_prefix = Output(OutChannel(self.var), self.actual_params)
        
        # Create the replication (!x(y).P)
        replication = Replication(
            PrefixProcess(
                Input(InChannel(self.var), self.params),
                self.proc
            )
        )
        
        # Create the parallel composition
        parallel = Parallel([
            PrefixProcess(output_prefix, Inaction()),
            replication
        ])
        
        return Restriction(
            names=[self.var],
            process=parallel
        )
        
    def __str__(self):
        params_str = f"({','.join(self.params)})" if self.params else ""
        actual_str = f"@<{','.join(self.actual_params)}>" if self.actual_params else ""
        return f"μ{self.var}{params_str}.{self.proc}{actual_str}"

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
        # 为Sum添加括号以匹配所需格式
        if len(self.branches) > 1:
            return f"({' + '.join(str(branch) for branch in self.branches)})"
        return str(self.branches[0]) if self.branches else "0"

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
    def __init__(self, condition: Expr, branch0: Process, branch1: Process):
        self.condition = condition
        self.branch0 = branch0
        self.branch1 = branch1
    
    def to_sum(self):
        # Convert to [B].P + [¬B].Q
        pos_guard = Guard(self.condition)
        neg_guard = Guard(UnaryOp("¬", self.condition))  # Assuming we have UnaryOp for negation
        
        pos_branch = PrefixProcess(pos_guard, self.branch0)
        neg_branch = PrefixProcess(neg_guard, self.branch1)
        
        return Sum([pos_branch, neg_branch])
    
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

class System(Process):
    def __init__(self, restricted_names: List[str], components: List[Process]):
        self.restricted_names = restricted_names
        self.components = components
        self.process = self._build_process()

    def _build_process(self) -> Restriction:
        parallel = Parallel(self.components)
        current_process: Process = parallel
        for name in reversed(self.restricted_names):
            current_process = Restriction([name], current_process)
        return current_process

    def __str__(self) -> str:
        return str(self.process)

# 测试代码
if __name__ == "__main__":
    # Train ≜ (ν p, v, a)(overline{channels}⟨p,v,a⟩ . Run || Observer)

    train_output = PrefixProcess(
        Output(OutChannel("channels"), ["p", "v", "a"]),
        Parallel([Inaction(), Inaction()])

    )

    train = Restriction(["p", "v", "a"], train_output)
    print(train)
