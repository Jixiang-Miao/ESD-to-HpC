#esd.py
import hpc
from typing import Dict, Set, Tuple, List
import re
from expr import Expr, Var

class Fragment:
    def __init__(self, assignment_tracker=None):
        self.cont = None
        self.assignment_tracker = assignment_tracker or AssignmentTracker()
        
    def _propagate_tracker(self):
        """Propagate the assignment tracker to continuation fragments"""
        if self.cont:
            self.cont.assignment_tracker = self.assignment_tracker
            if hasattr(self.cont, '_propagate_tracker'):
                self.cont._propagate_tracker()
                
    def translate(self, roles: List[str]):
        self._propagate_tracker()
        if self.cont:
            return self.cont.translate(roles)
        else:
            return {role: hpc.Inaction() for role in roles}
        
class AssignmentTracker:
    def __init__(self):
        self.assigned_vars = set()
        self.allocated_vars = []

    def is_first_assignment(self, var):
        if var not in self.assigned_vars:
            self.assigned_vars.add(var)
            self.allocated_vars.append(var)
            return True
        return False
class Assignment(Fragment):
    def __init__(self, role: str, var: str, expr: str, cont: Fragment):
        super().__init__()
        self.role = role
        self.var = var
        self.expr = expr
        self.cont = cont

    def translate(self, roles):
        ret = self.cont.translate(roles) if self.cont else {role: hpc.Inaction() for role in roles}
        assignment = hpc.Assignment(
            hpc.Var(self.var),
            hpc.Var(self.expr)
        )

        ret[self.role] = hpc.PrefixProcess(assignment, ret[self.role])
        return ret

class Communication(Fragment):
    def __init__(self, sender: str, receiver: str, channel: hpc.Channel,
                 var_list: List[str], expr_list: List[str], cont: Fragment, 
                 assignment_tracker=None, temporal: hpc.TemporalPrimitive = None):
        super().__init__(assignment_tracker)
        self.sender = sender
        self.receiver = receiver
        self.channel = channel
        self.var_list = var_list
        self.expr_list = expr_list
        self.cont = cont
        self.temporal = temporal

    def __repr__(self):
        return f"Communication(sender={self.sender}, receiver={self.receiver}, channel={self.channel}, temporal={self.temporal}, var_list={self.var_list},expr_list={self.expr_list})"

    def translate(self, roles):
        print(f"Translating Communication: sender={self.sender}, receiver={self.receiver}")
        assert self.sender in roles and self.receiver in roles

        ret = self.cont.translate(roles)
        output_channel = f"{self.channel.name}\u0305"
        sender_action = hpc.PrefixProcess(
            hpc.Output(hpc.OutChannel(output_channel, self.channel.para), self.expr_list),
            ret[self.sender]
        )
        receiver_action = hpc.PrefixProcess(
            hpc.Input(hpc.InChannel(self.channel.name, self.channel.para), self.var_list),
            ret[self.receiver]
        )
        if self.temporal:
            ret[self.sender] = add_temporal_constraint(sender_action, self.temporal)
            ret[self.receiver] = add_temporal_constraint(receiver_action, self.temporal)
        else:
            ret[self.sender] = sender_action
            ret[self.receiver] = receiver_action
        return ret

_temporal_var_counter = 0

def _fresh_temporal_var(prefix: str) -> str:
    global _temporal_var_counter
    _temporal_var_counter += 1
    return f"{prefix}_{_temporal_var_counter}"

def temporal_condition(temporal: hpc.TemporalPrimitive, t: str) -> str:
    kind = temporal.kind
    args = temporal.args

    if kind == "at":
        d = args[0]
        return f"{t} == {d}"
    if kind == "before":
        d = args[0]
        return f"{t} <= {d}"
    if kind == "after":
        d = args[0]
        return f"{t} >= {d}"
    if kind == "between":
        d0, d1 = args
        return f"{d0} <= {t} and {t} <= {d1}"
    if kind == "every":
        d = args[0]
        return f"{t} % {d} == 0"
    if kind == "periodic":
        d0, d1 = args
        return f"({t}-{d0}) % {d1} == 0"

    raise ValueError(f"Unsupported temporal primitive: {temporal}")

def add_temporal_constraint(action: hpc.Process, temporal: hpc.TemporalPrimitive) -> hpc.Process:
    t = _fresh_temporal_var("t")
    condition = temporal_condition(temporal, t)
    checked = hpc.PrefixProcess(hpc.Guard(condition), action.continuation)
    post_clock = hpc.PrefixProcess(hpc.Input(hpc.InChannel("clock"), [t]), checked)
    return hpc.PrefixProcess(action.prefix, post_clock)

class Sensation(Fragment):
    def __init__(self, sender: str, receiver: str, var_x: str, var_v: str, 
                 cont: Fragment, assignment_tracker=None):
        super().__init__(assignment_tracker)
        self.sender = sender
        self.receiver = receiver
        self.var_x = var_x
        self.var_v = var_v
        self.cont = cont

    def __repr__(self):
        return f"Sensation(sender={self.sender}, receiver={self.receiver}, var_x={self.var_x}, var_v={self.var_v})"
    
    def translate(self, roles):
        if self.cont is None:
            return {role: hpc.Inaction() for role in roles}
        else:
            assert self.sender in roles and self.receiver in roles
            ret = self.cont.translate(roles)
            ret[self.sender] = hpc.PrefixProcess(
                    hpc.Input(hpc.InChannel(self.var_v), [self.var_x]),
                    ret[self.sender]
                )
            return ret

class Actuation(Fragment):
    def __init__(self, sender: str, receiver: str, var_v: str, expr: str, cont: Fragment):
        self.sender = sender
        self.receiver = receiver
        self.var_v = var_v
        self.expr = expr
        self.cont = cont

    def __repr__(self):
        return f"Actuation(sender={self.sender}, receiver={self.receiver}, var_v={self.var_v}, expr={self.expr})"
    
    def translate(self, roles):
        assert self.sender in roles and self.receiver in roles
        ret = self.cont.translate(roles)
        ret[self.sender] = hpc.PrefixProcess(
            hpc.Output(hpc.OutChannel(f"{self.var_v}\u0305"), [self.expr]),
            ret[self.sender]
        )
        return ret

class Activation(Fragment):
    def __init__(self, role: str, ode: hpc.ODE, body: Fragment, cont: Fragment):
        self.role = role
        self.ode = ode
        self.body = body
        self.cont = cont

    def __repr__(self):
        return f"Activation(role={self.role}, ode={self.ode}, body={self.body})"
    
    def translate(self, roles):
        assert self.role in roles
        ready_set = get_ready_set(self.role, self.body)
        
        ode_vars = set(self.ode.v)
        filtered_ready_set = []
        for channel in ready_set:
            var_name = channel.name.replace('\u0305', '')
            if var_name in ode_vars:
                if isinstance(channel, hpc.InChannel) and channel.io == "in":
                    filtered_ready_set.append(hpc.InChannel(f"{var_name}"))
                else:
                    filtered_ready_set.append(channel)
        
        self.ode.ready_set = filtered_ready_set
        composed = compose(self.body, self.cont) if self.cont else self.body
        ret = composed.translate(roles)
        
        ret[self.role] = hpc.PrefixProcess(hpc.Continuous(self.ode), ret[self.role])
        return ret

_role_time_vars = {}
class Delay(Fragment):
    def __init__(self, role: str, delay: float, cont: Fragment):
        self.role = role
        self.delay = delay
        self.cont = cont

    def translate(self, roles):
        assert self.role in roles
        ret = self.cont.translate(roles)
        
        if self.role not in _role_time_vars:
            time_var = "t"
            _role_time_vars[self.role] = time_var
        else:
            time_var = _role_time_vars[self.role]
        
        wait_process = hpc.Wait(self.delay)
        restricted = wait_process.to_restriction(t_var=time_var)
        ret[self.role] = hpc.PrefixProcess(restricted, ret[self.role])
        return ret

class Alternative(Fragment):
    def __init__(self, index: int, role: str, cond: str, branch0: Fragment, branch1: Fragment, cont: Fragment):
        self.index = index
        self.role = role
        self.cond = cond
        self.branch0 = branch0
        self.branch1 = branch1
        self.cont = cont

    def translate(self, roles):
        """
        Convert the alt statement to the mixed π-calculus form.
        :param roles: Set of all participating roles
        :return: Transformed process definition
        """
        assert self.role in roles
        ret = {}

        ret0 = compose(self.branch0, self.cont).translate(roles)
        ret1 = compose(self.branch1, self.cont).translate(roles)

        roles0 = get_discrete_roles(self.branch0)
        roles1 = get_discrete_roles(self.branch1)
        all_roles = set(roles0).union(roles1)

        branch0 = ret0[self.role]
        branch1 = ret1[self.role]

        for role in all_roles:
            ch1 = f"alt1_{self.index}_{role}"
            ch2 = f"alt2_{self.index}_{role}"
            if role != self.role:
                branch0 = hpc.PrefixProcess(
                    hpc.Output(hpc.OutChannel(ch1)),
                    branch0
                )
                branch1 = hpc.PrefixProcess(
                    hpc.Output(hpc.OutChannel(ch2)),
                    branch1
                )
                ret[role] = hpc.Sum([
                    hpc.PrefixProcess(
                        hpc.Input(hpc.InChannel(ch1)),
                        ret0[role]
                    ),
                    hpc.PrefixProcess(
                        hpc.Input(hpc.InChannel(ch2)),
                        ret1[role]
                    )
                ])

        conditional = hpc.Conditional(self.cond, branch0, branch1)
        ret[self.role] = conditional.to_sum()

        for role in roles:
            if role == self.role:
                continue
            if role not in ret:
                ret[role] = ret0.get(role, hpc.Inaction())

        return ret

class Option(Fragment):
    def __init__(self, index: int, role: str, cond: str, body: Fragment, cont: Fragment):
        self.index = index
        self.role = role
        self.cond = cond
        self.body = body
        self.cont = cont

    def __repr__(self):
        return f"Option(index={self.index}, role={self.role}, cond={self.cond}, body={self.body})"
    
    def translate(self, roles):
        return Alternative(self.index, self.role, self.cond, self.body, Fragment(), self.cont).translate(roles)


def extract_condition_variables(cond: str) -> List[str]:
    if not cond:
        return []

    reserved = {"and", "or", "not", "True", "False"}
    tokens = re.findall(r"\b[a-zA-Z_]\w*\b", cond)
    ordered = []
    for token in tokens:
        if token in reserved:
            continue
        if token not in ordered:
            ordered.append(token)
    return ordered

def rewrite_break_in_loop(loop: "Loop") -> Fragment:
    break_var = f"break_{loop.index}"
    loop.cond = f"({loop.cond}) and {break_var} == 0"
    
    init_assign = Assignment(
        role=loop.role,
        var=break_var,
        expr="0",
        cont=None
    )
    
    visited = set()
    
    def transform(frag):
        if isinstance(frag, Fragment):
            frag.assignment_tracker = loop.assignment_tracker
        if frag is None:
            return None
        
        if id(frag) in visited:
            return frag
        visited.add(id(frag))
        
        if isinstance(frag, Break) and frag.in_loop:
            set_break = Assignment(
                role=frag.role,
                var=break_var,
                expr="1",
                cont=None
            )
            
            break_body = compose(frag.body, set_break) if frag.body else set_break
            
            normal_branch = frag.cont if frag.cont else Fragment()
            
            return Break(
                index=frag.index,
                role=frag.role,
                cond=frag.cond,
                body=break_body,
                normal_branch=normal_branch,
                cont=loop.cont,
                in_loop=True,
                loop_channel=f"loop[{loop.index}]",
                loop_actual_params=extract_condition_variables(loop.cond)
            )
        
        for attr in ["body", "branch0", "branch1", "frag0", "frag1"]:
            if hasattr(frag, attr):
                child = getattr(frag, attr)
                new_child = transform(child)
                setattr(frag, attr, new_child)
        
        if hasattr(frag, "cont") and frag.cont:
            frag.cont = transform(frag.cont)
        
        return frag
    
    loop.body = transform(loop.body)
    return compose(init_assign, loop)

class Loop(Fragment):
    def __init__(self, index: int, role: str, cond: str, body: Fragment, cont: Fragment):
        self.index = index
        self.role = role
        self.cond = cond
        self.body = body
        self.cont = cont
        self._break_rewritten = False

    def translate(self, roles):
        assert self.role in roles

        if not getattr(self, "_break_rewritten", False):
            if self._has_loop_break(self.body):
                rewritten = rewrite_break_in_loop(self)
                self._break_rewritten = True
                if rewritten is not self:
                    return rewritten.translate(roles)

        loop_channel = f"loop[{self.index}]"
        loop_params = extract_condition_variables(self.cond)

        body_proc = self.body.translate(roles)
        cont_proc = self.cont.translate(roles)

        ret = {}
        for role in roles:
            if role == self.role:
                cond = self.cond
                
                if self._has_loop_break(self.body):
                    then_proc = body_proc[role]
                else:
                    loop_call = hpc.PrefixProcess(
                        hpc.Output(hpc.OutChannel(loop_channel), loop_params),
                        hpc.Inaction()
                    )
                    then_proc = sequence(body_proc[role], loop_call)
                
                else_proc = cont_proc[role]

                conditional = hpc.Conditional(cond, then_proc, else_proc)
                recursion = hpc.Recursion(
                    var=loop_channel,
                    params=loop_params,
                    proc=conditional.to_sum(),
                    actual_params=loop_params
                )
                ret[role] = recursion.to_restriction()
            else:
                ret[role] = sequence(body_proc[role], cont_proc[role])

        return ret
    
    def _has_loop_break(self, frag):
        if frag is None:
            return False
        
        if isinstance(frag, Break) and frag.in_loop:
            return True
        
        for attr in ["body", "branch0", "branch1", "frag0", "frag1", "cont"]:
            if hasattr(frag, attr):
                if self._has_loop_break(getattr(frag, attr)):
                    return True
        
        return False
    
class Break(Fragment):
    def __init__(self, index, role, cond, body, normal_branch, cont, in_loop=False,
                 loop_channel=None, loop_actual_params=None):
        super().__init__()
        self.index = index
        self.role = role
        self.cond = cond
        self.body = body
        self.normal_branch = normal_branch
        self.cont = cont
        self.in_loop = in_loop
        self.loop_channel = loop_channel
        self.loop_actual_params = loop_actual_params or []

    def translate(self, roles):
        print(f"in_loop: {self.in_loop}")
        if self.in_loop:
            return self._translate_loop_break(roles)
        else:
            return self._translate_outer_break(roles)

    def _translate_outer_break(self, roles):
            ret_exceptional = self.body.translate(roles) if self.body else {r: hpc.Inaction() for r in roles}

            ret_normal = self.cont.translate(roles) if self.cont else {r: hpc.Inaction() for r in roles}
            
            ret = {}
            
            for role in roles:
                if role == self.role:
                    cond = self.cond
                    exceptional_branch = ret_exceptional[role]                    
                    normal_branch = ret_normal[role]
                    conditional = hpc.Conditional(
                        cond, 
                        exceptional_branch, 
                        normal_branch
                    )
                    ret[role] = conditional.to_sum()
                else:
                    except_ch = f"break_except_{self.index}_{role}"
                    normal_ch = f"break_normal_{self.index}_{role}"
                    
                    if not isinstance(ret_exceptional.get(role, hpc.Inaction()), hpc.Inaction):
                        exceptional_with_notify = hpc.PrefixProcess(
                            hpc.Output(hpc.OutChannel(except_ch), []),
                            ret_exceptional[role]
                        )
                    else:
                        exceptional_with_notify = hpc.Inaction()
                        
                    if not isinstance(ret_normal.get(role, hpc.Inaction()), hpc.Inaction):
                        normal_with_notify = hpc.PrefixProcess(
                            hpc.Output(hpc.OutChannel(normal_ch), []),
                            ret_normal[role]
                        )
                    else:
                        normal_with_notify = hpc.Inaction()
                    
                    ret[role] = hpc.Sum([
                      ret_exceptional.get(role, hpc.Inaction()
                        ),
                        ret_normal.get(role, hpc.Inaction())
                    ])
            
            return ret
    
    def _translate_loop_break(self, roles):
            ret_break = self.body.translate(roles) if self.body else {r: hpc.Inaction() for r in roles}
            ret_normal = self.normal_branch.translate(roles) if self.normal_branch else {r: hpc.Inaction() for r in roles}

            ret = {}
            loop_channel = self.loop_channel or f"loop[{self.index}]"
            loop_actual_params = self.loop_actual_params or extract_condition_variables(self.cond)

            for role in roles:
                if role == self.role:
                    loop_call = hpc.PrefixProcess(
                        hpc.Output(hpc.OutChannel(loop_channel), loop_actual_params),
                        hpc.Inaction()
                    )

                    exceptional_branch = sequence(ret_break.get(role, hpc.Inaction()), loop_call)
                    normal_branch = sequence(ret_normal.get(role, hpc.Inaction()), loop_call)

                    cond = hpc.Conditional(
                        self.cond,
                        exceptional_branch,
                        normal_branch
                    )
                    ret[role] = cond.to_sum()
                else:
                    br = ret_break.get(role, hpc.Inaction())
                    nr = ret_normal.get(role, hpc.Inaction())

                    def as_prefix_list(p):
                        if isinstance(p, hpc.Inaction):
                            return []
                        if isinstance(p, hpc.Sum):
                            return list(p.branches)
                        if isinstance(p, hpc.PrefixProcess):
                            return [p]
                        return [hpc.PrefixProcess(hpc.Tau(), p)]

                    branches = as_prefix_list(br) + as_prefix_list(nr)
                    if len(branches) == 0:
                        ret[role] = hpc.Inaction()
                    elif len(branches) == 1:
                        ret[role] = branches[0]
                    else:
                        ret[role] = hpc.Sum(branches)
            
            return ret

class Par(Fragment):
    def __init__(self, frag0: Fragment, frag1: Fragment, cont: Fragment):
        super().__init__()
        self.frag0 = frag0
        self.frag1 = frag1
        self.cont = cont

    def _as_sum_branches(self, p: hpc.Process) -> List[hpc.PrefixProcess]:
        if isinstance(p, hpc.Sum):
            return list(p.branches)
        if isinstance(p, hpc.PrefixProcess):
            return [p]
        
        return [hpc.PrefixProcess(hpc.Tau(), p)]

    def _choice(self, p: hpc.Process, q: hpc.Process) -> hpc.Process:
        b = self._as_sum_branches(p) + self._as_sum_branches(q)
        return b[0] if len(b) == 1 else hpc.Sum(b)

    def translate(self, roles):
        def get_first_step(frag, role):
            if frag is None:
                return None
            
            if isinstance(frag, Critical):
                return ("critical", frag)
            elif isinstance(frag, (Assignment, Communication, Sensation, Actuation)):
                return ("action", frag)
            elif isinstance(frag, Alternative):
                return ("choice", frag)
            elif isinstance(frag, (Loop, Option)):
                return ("structured", frag)
            elif isinstance(frag, Par):
                first0 = get_first_step(frag.frag0, role)
                first1 = get_first_step(frag.frag1, role)
                return ("par", (first0, first1))
            return None
        
        for role in roles:
            first0 = get_first_step(self.frag0, role)
            first1 = get_first_step(self.frag1, role)
            print(f"Par first steps for {role}: frag0={first0}, frag1={first1}")
        
        proc0 = self.frag0.translate(roles)
        proc1 = self.frag1.translate(roles)
        cont_proc = self.cont.translate(roles) if self.cont else {r: hpc.Inaction() for r in roles}

        ret = {}
        for role in roles:
            p0 = proc0.get(role, hpc.Inaction())
            p1 = proc1.get(role, hpc.Inaction())
            pc = cont_proc.get(role, hpc.Inaction())

            if isinstance(p0, hpc.Inaction):
                ret[role] = sequence(p1, pc)
                continue
            if isinstance(p1, hpc.Inaction):
                ret[role] = sequence(p0, pc)
                continue

            left_first = sequence(p0, sequence(p1, pc))
            right_first = sequence(p1, sequence(p0, pc))
            ret[role] = self._choice(left_first, right_first)

        return ret

class Critical(Fragment):
    def __init__(self, frag: Fragment, cont: Fragment):
        self.frag = frag
        self.cont = cont
    
    def __repr__(self):
        return f"Critical(frag={self.frag})"
    
    def translate(self, roles):
        if self.frag is None:
            cont_proc = self.cont.translate(roles) if self.cont else {r: hpc.Inaction() for r in roles}
            return cont_proc
        
        frag_proc = self.frag.translate(roles)
        
        cont_proc = self.cont.translate(roles) if self.cont else {r: hpc.Inaction() for r in roles}
        
        ret = {}
        for role in roles:
            critical_p = frag_proc.get(role, hpc.Inaction())
            continue_p = cont_proc.get(role, hpc.Inaction())
            ret[role] = sequence(critical_p, continue_p)
        
        return ret


def get_ready_set(role: str, frag: Fragment) -> List[hpc.Channel]:
    ready_set = []

    def collect(f: Fragment):
        if f is None:
            return
            
        if isinstance(f, Assignment):
            if f.role == role:
                ready_set.append(hpc.OutChannel(f.var))
        elif isinstance(f, Communication):
            if f.sender == role:
                ready_set.append(hpc.OutChannel(f.channel))
            elif f.receiver == role:
                ready_set.append(hpc.InChannel(f.channel))
        elif isinstance(f, Sensation):
            print(f"Sensation:{f.receiver},role:{role}")
            if f.receiver == role:
                ready_set.append(hpc.OutChannel(f"{f.var_v}\u0305"))
        elif isinstance(f, Actuation):
            print(f"Actuation:{f.sender},role:{role}")
            if f.sender != role:
                ready_set.append(hpc.InChannel(f.var_v))
        elif isinstance(f, Activation):
            if f.role == role:
                ready_set.extend(f.ode.ready_set)

            else:
                if f.body:
                    collect(f.body)

        elif isinstance(f, Alternative):
            if f.branch0:
                collect(f.branch0)
            if f.branch1:
                collect(f.branch1)
        elif isinstance(f, Option):
            if f.body:
                collect(f.body)
        elif isinstance(f, Loop):
            if f.body:
                collect(f.body)

        elif isinstance(f, Break):
            if f.body:
                collect(f.body)

        elif isinstance(f, Par):
            if f.frag0:
                collect(f.frag0)
            if f.frag1:
                collect(f.frag1)

        elif isinstance(f, Critical):
            if f.frag:
                collect(f.frag)

        if f.cont:
            collect(f.cont)

    collect(frag)

    unique = {}
    for ch in ready_set:
        key = (ch.name, getattr(ch, 'io', ''))
        if key not in unique:
            unique[key] = ch

    return list(unique.values())

def compose(frag0: Fragment, frag1: Fragment, seen=None) -> Fragment:
    if seen is None:
        seen = set()
    if frag0 is None:
        return frag1
    if id(frag0) in seen:
        raise ValueError("Circular reference detected in compose()")
    seen.add(id(frag0))
    
    if frag0 is frag1:
        return frag0
        
    if frag0.cont is None:
        frag0.cont = frag1
    else:
        frag0.cont = compose(frag0.cont, frag1, seen)
    return frag0

def get_discrete_roles(frag: Fragment) -> List[str]:
    roles = set()
    ode_roles = set()

    def collect(f: Fragment):
        if f is None:
            return
        if isinstance(f, Assignment):
            roles.add(f.role)
        elif isinstance(f, Communication):
            roles.update([f.sender, f.receiver])
        elif isinstance(f, Sensation):
            roles.add(f.receiver)
        elif isinstance(f, Actuation):
            roles.add(f.sender)
        elif isinstance(f, Activation):
            ode_roles.add(f.role)
            collect(f.body)
        elif isinstance(f, Alternative):
            collect(f.branch0)
            collect(f.branch1)
        elif isinstance(f, Option):
            collect(f.body)
        elif isinstance(f, Loop):
            collect(f.body)
        elif isinstance(f, Par):
            collect(f.frag0)
            collect(f.frag1)
        elif isinstance(f, Critical):
            collect(f.frag)
        elif isinstance(f, Break):
            collect(f.body)

        if f.cont:
            collect(f.cont)

    collect(frag)
    return list(roles)

def sequence(p1: hpc.Process, p2: hpc.Process) -> hpc.Process:
    if isinstance(p2, hpc.Inaction):
        return p1

    if isinstance(p1, hpc.Inaction):
        return p2

    if isinstance(p1, hpc.PrefixProcess):
        return hpc.PrefixProcess(p1.prefix, sequence(p1.continuation, p2))

    if isinstance(p1, hpc.Sum):
        new_branches = [
            hpc.PrefixProcess(b.prefix, sequence(b.continuation, p2))
            for b in p1.branches
        ]
        return hpc.Sum(new_branches) if new_branches else p2

    if isinstance(p1, hpc.Restriction):
        return hpc.Restriction(list(p1.names), sequence(p1.process, p2))

    if isinstance(p1, hpc.NamedProcess):
        return hpc.NamedProcess(p1.name, sequence(p1.body, p2))

    if isinstance(p1, hpc.Recursion):
        return p1
    if isinstance(p1, hpc.Replication):
        return p1
    if isinstance(p1, hpc.Parallel):
        return p1

    return p1

def generate_memory_processes(tracker):
    memory_procs = []
    for i, var in enumerate(tracker.allocated_vars):
        mem_proc = hpc.NamedProcess(
            name=f"Memory{i}",
            process=hpc.Replication(
                hpc.Input(hpc.InChannel(f"{var}"), "y", 
                    hpc.Sum([
                        hpc.Output(hpc.OutChannel(f"{var}"), "y", hpc.Output(hpc.OutChannel(f"{var}"), "y", hpc.Inaction())),
                        hpc.Input(hpc.InChannel(f"{var}"), "z", hpc.Output(hpc.OutChannel(f"{var}"), "z", hpc.Inaction()))
                    ])
                )
            )
        )
        memory_procs.append(mem_proc)
    return memory_procs


class ESD:
    def __init__(self, frag: Fragment, roles: List[str]):
        assert len(roles) >= 1
        self.frag = frag
        self.roles = roles

    def translate(self):
        ret = self.frag.translate(self.roles)

        named_procs = [hpc.NamedProcess(role, ret[role]) for role in self.roles]
        if has_temporal_communication(self.frag):
            named_procs.append(hpc.NamedProcess("GlobalClock", hpc.make_global_clock()))

        tracker = self.frag.assignment_tracker
        if hasattr(tracker, "memory_processes"):
            unique_mem_procs = []
            seen = set()
            for mem_proc in tracker.memory_processes:
                if mem_proc.name not in seen:
                    seen.add(mem_proc.name)
                    unique_mem_procs.append(mem_proc)
            named_procs.extend(unique_mem_procs)
            
        return hpc.Parallel(named_procs)

def parse_conditional_expr(expr_str):
    cond_match = re.match(r"(\w+)\s*([<>=!]+)\s*(\w+)", expr_str)
    if cond_match:
        left, op, right = cond_match.groups()
        return hpc.Conditional(left=left, op=op, right=right)
    else:
        raise ValueError(f"Unresolvable conditional expression: {expr_str}")

def set_tail_cont_to_fragment(head: Fragment):
    node = head
    while node and node.cont:
        node = node.cont
    if node:
        node.cont = Fragment()

def parse_channel_content(content: str):
    content = (content or "").strip()
    if not content:
        return [], []

    m = re.match(r'(.+?):=\s*(.+)', content)
    if m:
        left_vars = [v.strip() for v in m.group(1).split(',')]
        right_exprs = [e.strip() for e in m.group(2).split(',')]
    else:
        left_vars = [v.strip() for v in content.split(',')]
        right_exprs = left_vars.copy()

    if len(left_vars) != len(right_exprs):
        raise ValueError("Number of variables and expressions do not match!")

    return left_vars, right_exprs

def parse_temporal_primitive(tmp: str) -> hpc.TemporalPrimitive:
    tmp = tmp.strip()
    m = re.match(r"(at|before|after|every)\((\d+(?:\.\d+)?)\)$", tmp)
    if m:
        kind, d = m.groups()
        return hpc.TemporalPrimitive(kind, [float(d)])

    m = re.match(r"(between|periodic)\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)$", tmp)
    if m:
        kind, d0, d1 = m.groups()
        return hpc.TemporalPrimitive(kind, [float(d0), float(d1)])

    raise ValueError(f"Unsupported temporal primitive syntax: [{tmp}]")

def has_temporal_communication(frag: Fragment) -> bool:
    if frag is None:
        return False
    if isinstance(frag, Communication) and frag.temporal is not None:
        return True
    for attr in ["cont", "body", "branch0", "branch1", "frag0", "frag1", "frag"]:
        if hasattr(frag, attr) and has_temporal_communication(getattr(frag, attr)):
            return True
    return False

def parse_example(example: str) -> Fragment:
    lines = [line.strip() for line in example.strip().splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("@startuml") and not line.startswith("@enduml")]

    head = None
    tail = None
    tracker = AssignmentTracker()
    block_stack = []

    def append_top_level(node):
        nonlocal head, tail
        if head is None:
            head = tail = node
        else:
            tail.cont = node
            tail = node

    def append_node(node):
        if isinstance(node, Fragment):
            node.assignment_tracker = tracker

        if block_stack:
            top = block_stack[-1]
            t = top["type"]
            if t in ("opt", "loop", "break", "critical"):
                top["nodes"].append(node)
            elif t == "alt":
                if top["phase"] == "then":
                    top["then_nodes"].append(node)
                else:
                    top["else_nodes"].append(node)
            elif t == "par":
                top["branches"][top["active"]].append(node)
            elif t == "activation":
                top["body_nodes"].append(node)
            else:
                raise ValueError(f"Unknown block type: {t}")
        else:
            append_top_level(node)

    def link_nodes(nodes):
        if not nodes:
            return None
        for i in range(len(nodes) - 1):
            nodes[i].cont = nodes[i + 1]
        nodes[-1].cont = Fragment()
        print(f"Linked nodes: {[id(node) for node in nodes]}")
        return nodes[0]

    def close_top_block():
        if not block_stack:
            raise ValueError("Unexpected 'end': no open block")

        item = block_stack.pop()
        t = item["type"]

        if t == "loop":
            body = link_nodes(item["nodes"])
            node = Loop(
                index=item["index"],
                role=item["role"],
                cond=item["cond"],
                body=body if body else Fragment(),
                cont=Fragment()
            )
            append_node(node)
            return

        if t == "opt":
            body = link_nodes(item["nodes"])
            node = Option(
                index=item["index"],
                role=item["role"],
                cond=item["cond"],
                body=body if body else Fragment(),
                cont=Fragment()
            )
            append_node(node)
            return

        if t == "break":
            body = link_nodes(item["nodes"])
            in_loop = item.get("in_loop", False)
            node = Break(
                index=item["index"],
                role=item["role"],
                cond=item["cond"],
                body=body if body else Fragment(),
                normal_branch=Fragment(),
                cont=Fragment(),
                in_loop=in_loop
            )
            append_node(node)
            return

        if t == "alt":
            branch0 = link_nodes(item["then_nodes"]) if item["then_nodes"] else Fragment()
            branch1 = link_nodes(item["else_nodes"]) if item["else_nodes"] else Fragment()
            node = Alternative(
                index=item["index"],
                role=item["role"],
                cond=item["cond"],
                branch0=branch0,
                branch1=branch1,
                cont=Fragment()
            )
            append_node(node)
            return

        if t == "par":
            frag0 = link_nodes(item["branches"][0]) if item["branches"][0] else Fragment()
            frag1 = link_nodes(item["branches"][1]) if item["branches"][1] else Fragment()
            node = Par(frag0=frag0, frag1=frag1, cont=Fragment())
            append_node(node)
            return

        if t == "activation":
            activation_node = item["activation_node"]
            body = link_nodes(item["body_nodes"]) if item["body_nodes"] else None

            if activation_node:
                activation_node.body = body if body else Fragment()
                activation_node.ode.ready_set = get_ready_set(activation_node.role, activation_node.body)
                append_node(activation_node)
                return

            if body:
                append_node(body)
            return
        
        if t == "critical":
            body = link_nodes(item["nodes"])
            node = Critical(
                frag=body if body else Fragment(),
                cont=Fragment()
            )
            append_node(node)
            return

        raise ValueError(f"Unsupported block type on close: {t}")

    for line in lines:
        print(f"Parsed line: {line}")
        m = re.match(r"alt \[(\d+)\] (\w+): (.+)", line)
        if m:
            idx, role, cond = m.groups()
            block_stack.append({
                "type": "alt",
                "index": int(idx),
                "role": role,
                "cond": cond,
                "phase": "then",
                "then_nodes": [],
                "else_nodes": []
            })
            continue

        m = re.match(r"opt \[(\d+)\] (\w+): (.+)", line)
        if m:
            idx, role, cond = m.groups()
            block_stack.append({
                "type": "opt",
                "index": int(idx),
                "role": role,
                "cond": cond,
                "nodes": []
            })
            continue

        m = re.match(r"break \[(\d+)\] (\w+): (.+)", line)
        if m:
            idx, role, cond = m.groups()
            in_loop = any(b["type"] == "loop" for b in block_stack)
            block_stack.append({
                "type": "break",
                "index": int(idx),
                "role": role,
                "cond": cond,
                "in_loop": in_loop,
                "nodes": []
            })
            continue

        m = re.match(r"par \[(\d+)\]", line)
        if m:
            idx = m.group(1)
            block_stack.append({
                "type": "par",
                "index": int(idx),
                "active": 0,
                "branches": [[], []]
            })
            continue

        m = re.match(r"loop \[(\d+)\] (\w+): (.+)", line)
        if m:
            idx, role, cond = m.groups()
            block_stack.append({
                "type": "loop",
                "index": int(idx),
                "role": role,
                "cond": cond,
                "nodes": []
            })
            continue

        m = re.match(r"critical \[(\d+)\]", line)
        if m:
            idx = m.group(1)
            block_stack.append({
                "type": "critical",
                "index": int(idx),
                "nodes": []
            })
            continue

        if line.startswith("activate "):
            role = line.split()[1]
            block_stack.append({
                "type": "activation",
                "role": role,
                "activation_node": None,
                "body_nodes": []
            })
            continue

        if line == "else":
            if not block_stack:
                raise ValueError("Unexpected 'else': no open block")
            top = block_stack[-1]
            if top["type"] == "alt":
                top["phase"] = "else"
                continue
            if top["type"] == "par":
                top["active"] = 1
                continue
            raise ValueError(f"Unexpected 'else' in block type {top['type']}")

        else_m = re.match(r"else(?:\s+(.+))?", line)
        if else_m:
            if not block_stack or block_stack[-1]["type"] != "alt":
                raise ValueError("Unexpected 'else ...': current block is not alt")
            block_stack[-1]["phase"] = "else"
            continue

        if line == "end":
            close_top_block()
            continue

        delay_match = re.match(r"note (left|right|over) of (\w+): delay\((\d+(?:\.\d+)?)\)", line)
        if delay_match:
            _, role, delay_value = delay_match.groups()
            delay_node = Delay(role=role, delay=float(delay_value), cont=Fragment())
            append_node(delay_node)
            continue

        ode_match = re.match(r"note (left|right|over) of (\w+): <<ode>>\s*\{(.+)\}", line)
        if ode_match:
            _, role, ode_str = ode_match.groups()
            
            init_expr, rest = ode_str.split('|', 1)
            deriv_expr, bound = rest.split('&', 1)

            e0 = [v.strip() for v in init_expr.strip().split(',')]

            derivs = [v.strip() for v in deriv_expr.strip().split(',')]
            e = []
            v = []
            final_v = []

            for d in derivs:
                lhs, rhs = [x.strip() for x in d.split('=')]
                var = lhs.replace('_dot', '')
                v.append(var)
                e.append(rhs)
                final_v.append(f"{var}'")

            bound = bound.strip()

            ode = hpc.ODE(
                e0=e0,
                e=e,
                bound=bound,
                v=v,
                ready_set=None,
                final_v=final_v
            )

            if block_stack and block_stack[-1]["type"] == "activation":
                active_role = block_stack[-1].get("role")
                if active_role != role:
                    raise ValueError(
                        f"ODE role mismatch: ODE note role={role}, current activation role={active_role}"
                    )
                block_stack[-1]["activation_node"] = Activation(
                    role=role, ode=ode, body=None, cont=Fragment()
                )
            else:
                append_node(Activation(role=role, ode=ode, body=None, cont=Fragment()))
            continue

        if line.startswith("deactivate "):
            role = line.split()[1]
            if not block_stack or block_stack[-1]["type"] != "activation":
                raise ValueError("Unexpected deactivate without matching activate")
            if block_stack[-1].get("role") != role:
                raise ValueError(
                    f"Mismatched deactivate role: got {role}, expected {block_stack[-1].get('role')}"
                )
            close_top_block()
            continue

        assign_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*(\w+)\s*:=\s*(.+)", line)
        if assign_match:
            sender, receiver, var, expr = assign_match.groups()
            if sender == receiver:
                node = Assignment(role=sender, var=var, expr=expr.strip(), cont=Fragment())
                append_node(node)
                continue

        sense_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*<<sense>>\s*(\w+)\s*:=\s*(\w+)", line)
        if sense_match:
            sender, receiver, x, v = sense_match.groups()
            node = Sensation(sender=sender, receiver=receiver, var_x=x, var_v=v, cont=Fragment())
            append_node(node)
            continue


        act_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*<<actuate>>\s*(\w+)\s*[:=]+\s*(.+)", line)
        if act_match:
            sender, receiver, var, expr = act_match.groups()
            node = Actuation(sender=sender, receiver=receiver, var_v=var, expr=expr.strip(), cont=Fragment())
            append_node(node)
            continue

        temporal_comm_match = re.match(
            r"(\w+)\s*->\s*(\w+)\s*:\s*\[(.+?)\]\s*([A-Za-z_]\w*)\s*(?:\(([^()]*)\))?$",
            line
        )
        if temporal_comm_match:
            sender, receiver, tmp, channel_name, content = temporal_comm_match.groups()
            var_list, expr_list = parse_channel_content(content or "")
            node = Communication(
                sender=sender,
                receiver=receiver,
                channel=hpc.Channel(channel_name),
                var_list=var_list,
                expr_list=expr_list,
                cont=Fragment(),
                temporal=parse_temporal_primitive(tmp)
            )
            append_node(node)
            continue

        comm_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*([\w\s]+?)(?:\(([^()]*)\))?$", line)
        if comm_match:
            sender, receiver, channel_name, content = comm_match.groups()
            content = content.strip() if content else ""
            var_list, expr_list = parse_channel_content(content)

            node = Communication(
                sender=sender,
                receiver=receiver,
                channel=hpc.Channel(channel_name),
                var_list=var_list,
                expr_list=expr_list,
                cont=Fragment()
            )
            append_node(node)
            continue

        alt_comm_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*(\w+)", line)
        if alt_comm_match:
            sender, receiver, channel_name = alt_comm_match.groups()
            node = Communication(
                sender=sender,
                receiver=receiver,
                channel=hpc.Channel(channel_name),
                var_list=[],
                expr_list=[],
                cont=Fragment()
            )
            append_node(node)
            continue

    if block_stack:
        open_types = [b["type"] for b in block_stack]
        raise ValueError(f"Unclosed blocks at EOF: {open_types}")

    if head is not None:
        set_tail_cont_to_fragment(head)
        head.assignment_tracker = tracker
        if hasattr(head, "_propagate_tracker"):
            head._propagate_tracker()
    return head

def print_fragment(fragment, indent=0):
    prefix = "  " * indent
    current = fragment
    while current:
        print(f"{prefix}{current}")
        print(f"{prefix}{current}")
        if hasattr(current, "body") and current.body:
            print_fragment(current.body, indent + 1)
        current = getattr(current, "cont", None)


def print_nested_structure(fragment, indent=0, seen=None):
    if fragment is None:
        return
    
    if seen is None:
        seen = set()
    
    frag_id = id(fragment)
    if frag_id in seen:
        print("  " * indent + f"[loop reference: {type(fragment).__name__}]")
        return
    seen.add(frag_id)
    
    prefix = "  " * indent
    frag_type = type(fragment).__name__
    
    details = []
    if isinstance(fragment, Assignment):
        details = [f"role={fragment.role}", f"var={fragment.var}", f"expr={fragment.expr}"]
    elif isinstance(fragment, Communication):
        details = [f"sender={fragment.sender}", f"receiver={fragment.receiver}", 
                  f"channel={fragment.channel.name}"]
    elif isinstance(fragment, Sensation):
        details = [f"sender={fragment.sender}", f"receiver={fragment.receiver}",
                  f"var_x={fragment.var_x}", f"var_v={fragment.var_v}"]
    elif isinstance(fragment, Actuation):
        details = [f"sender={fragment.sender}", f"receiver={fragment.receiver}",
                  f"var_v={fragment.var_v}", f"expr={fragment.expr}"]
    elif isinstance(fragment, Activation):
        details = [f"role={fragment.role}", f"ode_vars={fragment.ode.v}"]
    elif isinstance(fragment, Delay):
        details = [f"role={fragment.role}", f"delay={fragment.delay}"]
    elif isinstance(fragment, Alternative):
        details = [f"index={fragment.index}", f"role={fragment.role}", f"cond={fragment.cond}"]
    elif isinstance(fragment, Option):
        details = [f"index={fragment.index}", f"role={fragment.role}", f"cond={fragment.cond}"]
    elif isinstance(fragment, Loop):
        details = [f"index={fragment.index}", f"role={fragment.role}", f"cond={fragment.cond}"]
    elif isinstance(fragment, Par):
        details = ["parallel fragment"]
    elif isinstance(fragment, Critical):
        details = ["critical section fragment"]
    elif isinstance(fragment, Break):
        details = ["break fragment"]
    
    print(f"{prefix}{frag_type}({', '.join(details)})")
    
    next_indent = indent + 1
    
    if hasattr(fragment, 'body') and fragment.body is not None:
        print(f"{prefix}  body:")
        print_nested_structure(fragment.body, next_indent, seen)
    
    if isinstance(fragment, Alternative):
        print(f"{prefix}  branch0:")
        print_nested_structure(fragment.branch0, next_indent, seen)
        print(f"{prefix}  branch1:")
        print_nested_structure(fragment.branch1, next_indent, seen)
    
    if isinstance(fragment, Par):
        print(f"{prefix}  frag0:")
        print_nested_structure(fragment.frag0, next_indent, seen)
        print(f"{prefix}  frag1:")
        print_nested_structure(fragment.frag1, next_indent, seen)
    
    if isinstance(fragment, Critical) and fragment.frag is not None:
        print(f"{prefix}  critical_frag:")
        print_nested_structure(fragment.frag, next_indent, seen)
    
    if isinstance(fragment, Break) and fragment.frag is not None:
        print(f"{prefix}  break_frag:")
        print_nested_structure(fragment.frag, next_indent, seen)
    
    if hasattr(fragment, 'cont') and fragment.cont is not None:
        print(f"{prefix}  cont:")
        print_nested_structure(fragment.cont, next_indent, seen)

def collect_role_variables(root_frag: Fragment) -> dict:
    role_vars = {}
    
    def init_role(role: str):
        if role not in role_vars:
            role_vars[role] = {
                'assigned': set(),
                'delay': set(),
                'ode': set(),
                'sent': set(),
                'received': set(),
                'sensed': set(),
                'actuated': set()
            }
    
    def _collect(frag: Fragment):
        if not frag:
            return
            
        if isinstance(frag, Assignment):
            init_role(frag.role)
            role_vars[frag.role]['assigned'].add(frag.var)
            expr_vars = re.findall(r'\b\w+\b', frag.expr)
            for var in expr_vars:
                if var not in [frag.var, frag.role]:
                    role_vars[frag.role]['sent'].add(var)

        elif isinstance(frag, Delay):
            init_role(frag.role)
            time_var = f"t_{frag.role}_{id(frag)}"
            role_vars[frag.role]['delay'].add(time_var)
        
        elif isinstance(frag, Communication):
            init_role(frag.sender)
            init_role(frag.receiver)
            for expr in frag.expr_list:
                expr_vars = re.findall(r'\b\w+\b', expr)
                for var in expr_vars:
                    if var != frag.sender:
                        role_vars[frag.sender]['sent'].add(var)
            for var in frag.var_list:
                role_vars[frag.receiver]['received'].add(var)
        
        elif isinstance(frag, Sensation):
            init_role(frag.receiver)
            role_vars[frag.receiver]['sensed'].add(frag.var_x)
            role_vars[frag.receiver]['received'].add(frag.var_v)
        
        elif isinstance(frag, Actuation):
            init_role(frag.sender)
            expr_vars = re.findall(r'\b\w+\b', frag.expr)
            for var in expr_vars:
                if var != frag.sender:
                    role_vars[frag.sender]['actuated'].add(var)
            role_vars[frag.sender]['sent'].add(frag.var_v)
        
        elif isinstance(frag, Activation):
            init_role(frag.role)
            for var in frag.ode.v:
                role_vars[frag.role]['ode'].add(var)
            _collect(frag.body)
        
        elif isinstance(frag, Loop):
            init_role(frag.role)
            cond_vars = re.findall(r'\b\w+\b', frag.cond)
            for var in cond_vars:
                if var != frag.role:
                    role_vars[frag.role]['received'].add(var)
            _collect(frag.body)
        
        elif isinstance(frag, Alternative):
            init_role(frag.role)
            cond_vars = re.findall(r'\b\w+\b', frag.cond)
            for var in cond_vars:
                if var != frag.role:
                    role_vars[frag.role]['received'].add(var)
            _collect(frag.branch0)
            _collect(frag.branch1)
        
        elif isinstance(frag, Par):
            _collect(frag.frag0)
            _collect(frag.frag1)
        
        if hasattr(frag, 'cont'):
            _collect(frag.cont)
    
    _collect(root_frag)
    return role_vars

def rename_conflicting_variables(root_frag: Fragment, role_vars: Dict[str, Dict[str, Set[str]]]) -> None:
    all_vars: Dict[str, Set[str]] = {}
    for role, vars_dict in role_vars.items():
        for var in vars_dict['assigned']:
            if var not in all_vars:
                all_vars[var] = set()
            all_vars[var].add(role)
    
    conflict_vars = {var: roles for var, roles in all_vars.items() if len(roles) > 1}
    if not conflict_vars:
        return
    
    var_rename_map: Dict[str, Dict[str, str]] = {}
    for var, roles in conflict_vars.items():
        rename_map = {}
        for i, role in enumerate(roles, 1):
            new_name = f"{var}_{i}"
            while any(new_name in role_vars[r]['assigned'] for r in role_vars if r != role):
                i += 1
                new_name = f"{var}_{i}"
            rename_map[role] = new_name
            role_vars[role]['assigned'].remove(var)
            role_vars[role]['assigned'].add(new_name)
        var_rename_map[var] = rename_map
    
    _rename_fragment_variables(root_frag, var_rename_map)

def _rename_fragment_variables(frag: Fragment, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    if not frag:
        return
    
    if isinstance(frag, Assignment):
        _rename_assignment_vars(frag, var_rename_map)
    elif isinstance(frag, Communication):
        _rename_communication_vars(frag, var_rename_map)
    elif isinstance(frag, Sensation):
        _rename_sensation_vars(frag, var_rename_map)
    elif isinstance(frag, Actuation):
        _rename_actuation_vars(frag, var_rename_map)
    elif isinstance(frag, Activation):
        _rename_activation_vars(frag, var_rename_map)
    elif isinstance(frag, Alternative):
        _rename_alternative_vars(frag, var_rename_map)
    elif isinstance(frag, Option):
        _rename_option_vars(frag, var_rename_map)
    elif isinstance(frag, Loop):
        _rename_loop_vars(frag, var_rename_map)
    
    if hasattr(frag, 'body'):
        _rename_fragment_variables(frag.body, var_rename_map)
    if isinstance(frag, Alternative):
        _rename_fragment_variables(frag.branch0, var_rename_map)
        _rename_fragment_variables(frag.branch1, var_rename_map)
    if isinstance(frag, Par):
        _rename_fragment_variables(frag.frag0, var_rename_map)
        _rename_fragment_variables(frag.frag1, var_rename_map)
    if isinstance(frag, Critical):
        _rename_fragment_variables(frag.frag, var_rename_map)
    if isinstance(frag, Break):
        _rename_fragment_variables(frag.frag, var_rename_map)
    
    if hasattr(frag, 'cont'):
        _rename_fragment_variables(frag.cont, var_rename_map)

def _rename_assignment_vars(assign: Assignment, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    for var, role_map in var_rename_map.items():
        if var == assign.var and assign.role in role_map:
            assign.var = role_map[assign.role]
    
    assign.expr = _rename_expr_vars(assign.expr, assign.role, var_rename_map)

def _rename_communication_vars(comm: Communication, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    for i, var in enumerate(comm.var_list):
        for orig_var, role_map in var_rename_map.items():
            if var == orig_var and comm.receiver in role_map:
                comm.var_list[i] = role_map[comm.receiver]
    
    for i, expr in enumerate(comm.expr_list):
        comm.expr_list[i] = _rename_expr_vars(expr, comm.sender, var_rename_map)

def _rename_sensation_vars(sense: Sensation, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    for var, role_map in var_rename_map.items():
        if var == sense.var_x and sense.receiver in role_map:
            sense.var_x = role_map[sense.receiver]
        if var == sense.var_v and sense.sender in role_map:
            sense.var_v = role_map[sense.sender]

def _rename_actuation_vars(act: Actuation, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    for var, role_map in var_rename_map.items():
        if var == act.var_v and act.sender in role_map:
            act.var_v = role_map[act.sender]
    
    act.expr = _rename_expr_vars(act.expr, act.sender, var_rename_map)

def _rename_activation_vars(activation: Activation, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    role = activation.role
    for i, var in enumerate(activation.ode.v):
        for orig_var, role_map in var_rename_map.items():
            if var == orig_var and role in role_map:
                activation.ode.v[i] = role_map[role]
    

    activation.ode.bound = _rename_expr_vars(activation.ode.bound, role, var_rename_map)

def _rename_alternative_vars(alt: Alternative, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    alt.cond = _rename_expr_vars(alt.cond, alt.role, var_rename_map)

def _rename_option_vars(option: Option, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    option.cond = _rename_expr_vars(option.cond, option.role, var_rename_map)

def _rename_loop_vars(loop: Loop, var_rename_map: Dict[str, Dict[str, str]]) -> None:
    loop.cond = _rename_expr_vars(loop.cond, loop.role, var_rename_map)

def _rename_expr_vars(expr: str, role: str, var_rename_map: Dict[str, Dict[str, str]]) -> str:
    if not expr:
        return expr
    
    vars_in_expr = re.findall(r'\b\w+\b', expr)
    
    for var in vars_in_expr:
        if var in var_rename_map and role in var_rename_map[var]:
            new_var = var_rename_map[var][role]
            expr = re.sub(rf'\b{var}\b', new_var, expr)
    
    return expr

def main():
    from standardize_process import to_standard_form
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()

    root = parse_example(example)
    print("=== Parsing Result ===")
    print_fragment(root)
    
    role_variables = collect_role_variables(root)
    print("\n=== Role Variables Collection Result ===")
    for role, var_types in role_variables.items():
        print(f"\nRole: {role}")
        for var_type, vars_set in var_types.items():
            if vars_set:
                print(f"  {var_type} variables: {sorted(vars_set)}")
    
    roles = ["Train", "LeftSector", "RightSector"]
    role_vars = collect_role_variables(root)
    rename_conflicting_variables(root, role_vars)
    esd = ESD(root, roles)
    translated = esd.translate()

    print("\n=== Conversion Result ===")
    print(translated)

    with open("example_translated_output.txt", "w", encoding="utf-8") as f:
        f.write(str(translated))
        f.write("\n")
    standard_form = to_standard_form(translated)
    print("\n=== Standard Form Result ===")
    print(standard_form)

    with open("example_standardized_output.txt", "w", encoding="utf-8") as f:
        f.write(str(standard_form))
    print("Conversion result saved")

if __name__ == "__main__":
    main()