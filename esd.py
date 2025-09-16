#esd.py
from standardize_process import to_standard_form
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
    def __init__(self, role: str, var: str, expr: str, cont: Fragment, assignment_tracker=None):
        super().__init__(assignment_tracker)
        self.role = role
        self.var = var
        self.expr = expr
        self.cont = cont
    def translate(self, roles: List[str]):
        self._propagate_tracker()
        assert self.role in roles
        ret = self.cont.translate(roles) if self.cont else {role: hpc.Inaction() for role in roles}

        is_ode_var = False
        if hasattr(self.cont, 'ode') and self.var in getattr(self.cont.ode, 'v', []):
            is_ode_var = True

        if is_ode_var:
            assignment = hpc.Assignment(hpc.Var(self.var), hpc.Var(self.expr))
            prefix = assignment.to_restriction(is_first=False)
            ret[self.role] = hpc.PrefixProcess(prefix, ret[self.role])
            print(f"Assign {self.var}: is_first={is_first}, is_ode_var={is_ode_var}")
        else:
            is_first = self.assignment_tracker.is_first_assignment(self.var)
            print(f"Assign {self.var}: is_first={is_first}, is_ode_var={is_ode_var}")
            assignment = hpc.Assignment(hpc.Var(self.var), hpc.Var(self.expr))
            prefix = assignment.to_restriction(is_first=is_first)
            ret[self.role] = hpc.PrefixProcess(prefix, ret[self.role])

            if is_first:
                mem_index = len(self.assignment_tracker.allocated_vars) - 1

                # !allocation(var)(y).(store(var)<y>.allocation(var)<y> + load(var)(z).allocation(var)<z>)
                memory_proc = hpc.NamedProcess(
                    f"Memory{mem_index}",
                    hpc.Replication(
                        hpc.PrefixProcess(
                            hpc.Input(hpc.InChannel(f"{self.var}"), ["y"]),
                            hpc.Sum([
                                hpc.PrefixProcess(
                                    hpc.Output(hpc.OutChannel(f"store({self.var}\u0305)"), ["y"]),
                                    hpc.PrefixProcess(
                                        hpc.Output(hpc.OutChannel(f"allocation({self.var})"), ["y"]),
                                        hpc.Inaction()
                                    )
                                ),
                                hpc.PrefixProcess(
                                    hpc.Input(hpc.InChannel(f"load({self.var})"), ["z"]),
                                    hpc.PrefixProcess(
                                        hpc.Output(hpc.OutChannel(f"allocation({self.var})"), ["z"]),
                                        hpc.Inaction()
                                    )
                                )
                            ])
                        )
                    )
                )

                if not hasattr(self.assignment_tracker, "memory_processes"):
                    self.assignment_tracker.memory_processes = []
                self.assignment_tracker.memory_processes.append(memory_proc)
                print(f"Assign {self.var}: is_first={is_first}, is_ode_var={is_ode_var}")

        return ret
    
class Communication(Fragment):
    def __init__(self, sender: str, receiver: str, channel: hpc.Channel,
                 var_list: List[str], expr_list: List[str], cont: Fragment, 
                 assignment_tracker=None):
        super().__init__(assignment_tracker)
        self.sender = sender
        self.receiver = receiver
        self.channel = channel
        self.var_list = var_list
        self.expr_list = expr_list
        self.cont = cont

    def __repr__(self):
        return f"Communication(sender={self.sender}, receiver={self.receiver}, channel={self.channel}, var_list={self.var_list},expr_list={self.expr_list})"

    def translate(self, roles):
        assert self.sender in roles and self.receiver in roles

        ret = self.cont.translate(roles)
        output_channel = f"{self.channel.name}\u0305"
        ret[self.sender] = hpc.PrefixProcess(
            hpc.Output(hpc.OutChannel(output_channel, self.channel.para), self.expr_list),
            ret[self.sender]
        )
        ret[self.receiver] = hpc.PrefixProcess(
            hpc.Input(hpc.InChannel(self.channel.name, self.channel.para), self.var_list),
            ret[self.receiver]
        )
        return ret



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
        var_v_with_macron = f"{self.var_v}\u0305"
        ret[self.sender] = hpc.PrefixProcess(
            hpc.Output(hpc.OutChannel(var_v_with_macron), [self.expr]),
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

class Delay(Fragment):
    def __init__(self, role: str, delay: float, cont: Fragment):
        self.role = role
        self.delay = delay
        self.cont = cont

    def translate(self, roles):
        assert self.role in roles
        ret = self.cont.translate(roles)
        
        time_var = f"t_{self.role}_{id(self)}"
        
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
        assert self.role in roles
        ret = dict()
        ret0 = compose(self.branch0, self.cont).translate(roles)
        ret1 = compose(self.branch1, self.cont).translate(roles)
        branch0 = ret0[self.role]
        branch1 = ret1[self.role]
        roles0 = get_discrete_roles(self.branch0)
        roles1 = get_discrete_roles(self.branch1)
        assert self.role in roles0 + roles1 and all(role in roles for role in roles0 + roles1)
        for role in roles0 + roles1:
            if role != self.role:
                branch0 = hpc.PrefixProcess(hpc.Output(hpc.OutChannel("alt", [str(self.index), role])), branch0)
                branch1 = hpc.PrefixProcess(hpc.Output(hpc.OutChannel("alt'", [str(self.index), role])), branch1)
                ret[role] = hpc.Sum([
                        hpc.PrefixProcess(hpc.Input(hpc.InChannel("alt", [str(self.index), role])), ret0[role]),
                        hpc.PrefixProcess(hpc.Input(hpc.InChannel("alt'", [str(self.index), role])), ret1[role])
                    ])
        
        # Convert the conditional to a sum
        conditional = hpc.Conditional(self.cond, branch0, branch1)
        ret[self.role] = conditional.to_sum()
        
        for role in roles:
            if role not in roles0 + roles1:
                ret[role] = ret0[role]  # assuming equal
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


class Loop(Fragment):
    def __init__(self, index: int, role: str, cond: str, body: Fragment, cont: Fragment):
        self.index = index
        self.role = role
        self.cond = cond
        self.body = body
        self.cont = cont

    def translate(self, roles):
        assert self.role in roles
        loop_index = self.index
        loop_channel = f"loop[{self.index}]"  # 添加横线
        loop_body_frag = self.body
        loop_body_proc = loop_body_frag.translate(roles)
        cont_proc = self.cont.translate(roles)

        ret = {}
        for role in roles:
            if role == self.role:
                cond = self.cond
                body_proc = loop_body_proc[role]
                loop_call = hpc.PrefixProcess(
                    hpc.Output(hpc.OutChannel(loop_channel), []),
                    hpc.Inaction()
                )
                then_proc = sequence(body_proc, loop_call)
                else_proc = cont_proc[role]
                
                # Convert conditional to sum
                conditional = hpc.Conditional(cond, then_proc, else_proc)
                
                # Create recursion and convert to restriction
                recursion = hpc.Recursion(
                    var=loop_channel,
                    params=[],
                    proc=conditional.to_sum(),
                    actual_params=[]
                )
                ret[role] = recursion.to_restriction()
            else:
                ret[role] = sequence(loop_body_proc[role], cont_proc[role])
        return ret

class Break(Fragment):
    def __init__(self, frag: Fragment):
        super().__init__()
        self.frag = frag


class Par(Fragment):
    def __init__(self, frag0: Fragment, frag1: Fragment, cont: Fragment):
        self.frag0 = frag0
        self.frag1 = frag1
        self.cont = cont

    def translate(self, roles):
        proc0 = self.frag0.translate(roles)
        proc1 = self.frag1.translate(roles)
        cont_proc = self.cont.translate(roles)
        ret = {}
        for role in roles:
            ret[role] = hpc.Parallel(proc0.get(role, hpc.Inaction()), proc1.get(role, hpc.Inaction()))
            ret[role] = hpc.Parallel(ret[role], cont_proc.get(role, hpc.Inaction()))
        return ret


class Critical(Fragment):
    def __init__(self, frag: Fragment, cont: Fragment):
        self.frag = frag
        self.cont = cont


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
        # Alternative（alt）
        elif isinstance(f, Alternative):
            if role != f.role:
                if f.branch0:
                    collect(f.branch0)
                if f.branch1:
                    collect(f.branch1)
            else:
                pass
        elif isinstance(f, Option):
            if role != f.role and f.body:
                collect(f.body)
        elif isinstance(f, Loop):
            if role != f.role and f.body:
                collect(f.body)

        elif isinstance(f, Break):
            if f.frag:
                collect(f.frag)

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
            collect(f.frag)

        if f.cont:
            collect(f.cont)

    collect(frag)
    return list(roles - ode_roles)

def sequence(p1: hpc.Process, p2: hpc.Process) -> hpc.Process:
    if isinstance(p1, hpc.Inaction):
        return p2
    elif isinstance(p1, hpc.PrefixProcess):
        return hpc.PrefixProcess(p1.prefix, sequence(p1.continuation, p2))
    else:
        return hpc.PrefixProcess(hpc.Tau(), sequence(p1, p2))

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
        assert len(roles) >= 2
        self.frag = frag
        self.roles = roles

    def translate(self):
        ret = self.frag.translate(self.roles)

        named_procs = [hpc.NamedProcess(role, ret[role]) for role in self.roles]

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
        raise ValueError(f"无法解析的条件表达式: {expr_str}")

def set_tail_cont_to_fragment(head: Fragment):
    node = head
    while node and node.cont:
        node = node.cont
    if node:
        node.cont = Fragment()

def parse_channel_content(content: str):
    m = re.match(r'([\w\s,]+):=\s*([\w\s,]+)', content)
    if m:
        left_vars = [v.strip() for v in m.group(1).split(',')]
        right_exprs = [e.strip() for e in m.group(2).split(',')]
    else:
        left_vars = [v.strip() for v in content.split(',')]
        right_exprs = left_vars.copy()

    if len(left_vars) != len(right_exprs):
        raise ValueError("Number of variables and expressions do not match!")

    return left_vars, right_exprs

def parse_example(example: str) -> Fragment:
    lines = [line.strip() for line in example.strip().splitlines() if line.strip()]
    lines = [line for line in lines if not line.startswith("@startuml") and not line.startswith("@enduml")]

    head = None
    tail = None
    pending_activation = None
    loop_stack = []

    activation_stack = []

    def append_node(node):
        nonlocal head, tail
        tracker = AssignmentTracker()
        # When creating the first fragment, pass the tracker
        if isinstance(node, Fragment):
            node.assignment_tracker = tracker
        if loop_stack:
            loop_stack[-1]['nodes'].append(node)
        elif activation_stack:
            activation_stack[-1]['body_nodes'].append(node)
        else:
            if head is None:
                head = tail = node
            else:
                tail.cont = node
                tail = node

    def link_nodes(nodes):
        if not nodes:
            return None
        for i in range(len(nodes) - 1):
            nodes[i].cont = nodes[i + 1]
        nodes[-1].cont = Fragment()
        return nodes[0]

    for line in lines:
        # print(f"pending_activation:{pending_activation}")
        # === 1. loop  ===
        loop_match = re.match(r"loop \[(\d+)\] (\w+): (.+)", line)
        if loop_match:
            idx, role, cond = loop_match.groups()
            loop_stack.append({'index': int(idx), 'role': role, 'cond': cond, 'nodes': []})
            continue
        if line == "end" and loop_stack:
            loop = loop_stack.pop()
            body_head = link_nodes(loop['nodes'])
            loop_node = Loop(index=loop['index'], role=loop['role'], cond=loop['cond'], body=body_head, cont=Fragment())
            append_node(loop_node)
            continue

        # === 2. activate ===
        if line.startswith("activate "):
            role = line.split()[1]
            activation_stack.append({'role': role, 'body_nodes': [], 'activation_node': None})
            continue

        # === 3. note  (delay(x)) ===
        delay_match = re.match(r"note (left|right|over) of (\w+): delay\((\d+(?:\.\d+)?)\)", line)
        if delay_match:
            _, role, delay_value = delay_match.groups()
            delay_node = Delay(role=role, delay=float(delay_value), cont=Fragment())
            append_node(delay_node)
            continue


        # === 3b. note  (<<ode>> {...}) ===
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

            if activation_stack:
                activation = Activation(role=role, ode=ode, body=None, cont=Fragment())
                activation_stack[-1]['activation_node'] = activation
            else:
                append_node(Activation(role=role, ode=ode, body=None, cont=Fragment()))
            continue

        if line.startswith("deactivate "):
            if activation_stack:
                item = activation_stack.pop()
                activation_node = item['activation_node']
                if activation_node:
                    activation_node.body = link_nodes(item['body_nodes'])
                    
                    activation_node.ode.ready_set = get_ready_set(activation_node.role, activation_node.body)

                    append_node(activation_node)
            continue

        # === 5. Assignment (:=) ===
        assign_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*(\w+)\s*:=\s*(.+)", line)
        if assign_match:
            sender, receiver, var, expr = assign_match.groups()
            if sender == receiver:
                node = Assignment(role=sender, var=var, expr=expr.strip(), cont=Fragment())
                append_node(node)
                continue

        # === 6. <<sense>> ===
        sense_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*<<sense>>\s*(\w+)\s*:=\s*(\w+)", line)
        if sense_match:
            sender, receiver, x, v = sense_match.groups()
            node = Sensation(sender=sender, receiver=receiver, var_x=x, var_v=v, cont=Fragment())
            append_node(node)
            continue


        # === 7. <<actuate>> ===
        act_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*<<actuate>>\s*(\w+)\s*[:=]+\s*(.+)", line)
        if act_match:
            sender, receiver, var, expr = act_match.groups()
            node = Actuation(sender=sender, receiver=receiver, var_v=var, expr=expr.strip(), cont=Fragment())
            append_node(node)
            continue

        # === 8. channels(...)  ===
        comm_match = re.match(r"(\w+)\s*->\s*(\w+)\s*:\s*(\w+)\(([^()]*)\)", line)
        if comm_match:
            sender, receiver, channel_name, content = comm_match.groups()
            content = content.strip()
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


        # === 9. comm ===
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
    if head is not None:
        set_tail_cont_to_fragment(head)
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
        print("  " * indent + f"[循环引用: {type(fragment).__name__}]")
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
        details = ["并行片段"]
    elif isinstance(fragment, Critical):
        details = ["临界区片段"]
    elif isinstance(fragment, Break):
        details = ["中断片段"]
    
    # 打印当前片段信息
    print(f"{prefix}{frag_type}({', '.join(details)})")
    
    # 递归处理子结构
    next_indent = indent + 1
    
    # 处理body属性（如Activation, Loop等）
    if hasattr(fragment, 'body') and fragment.body is not None:
        print(f"{prefix}  body:")
        print_nested_structure(fragment.body, next_indent, seen)
    
    # 处理分支结构（如Alternative）
    if isinstance(fragment, Alternative):
        print(f"{prefix}  branch0:")
        print_nested_structure(fragment.branch0, next_indent, seen)
        print(f"{prefix}  branch1:")
        print_nested_structure(fragment.branch1, next_indent, seen)
    
    # 处理并行结构（Par）
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
        """初始化角色的变量字典"""
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
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()

    root = parse_example(example)
    print("=== 解析结果 ===")
    print_fragment(root)
    
    role_variables = collect_role_variables(root)
    print("\n=== 各角色变量收集结果 ===")
    for role, var_types in role_variables.items():
        print(f"\n角色: {role}")
        for var_type, vars_set in var_types.items():
            if vars_set:
                print(f"  {var_type}变量: {sorted(vars_set)}")
    
    roles = ["Train", "LeftSector", "RightSector"]
    
    role_vars = collect_role_variables(root)
    rename_conflicting_variables(root, role_vars)
    esd = ESD(root, roles)
    translated = esd.translate()

    print("\n=== 转换结果 ===")
    print(translated)

    with open("translated_output.txt", "w", encoding="utf-8") as f:
        f.write(str(translated))
        f.write("\n")
    standard_form = to_standard_form(translated)
    print("\n=== 标准型结果 ===")
    print(standard_form)

    with open("standardized_output.txt", "w", encoding="utf-8") as f:
        f.write(str(standard_form))
    print("转换结果已保存")

if __name__ == "__main__":
    main()