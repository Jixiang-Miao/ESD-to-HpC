#esd.py
import hpc
from typing import List, Set
import re


class Fragment:
    def __init__(self):
        self.cont = None

    def __repr__(self):
        return f""
    
    def translate(self, roles: List[str]):
        return {role:hpc.Inaction() for role in roles}


class Assignment(Fragment):
    def __init__(self, role: str, var: str, expr: str, cont: Fragment):
        self.role = role
        self.var = var
        self.expr = expr
        self.cont = cont

    def __repr__(self):
        return f"Assignment(role={self.role}, var={self.var}, expr={self.expr})"
    
    def translate(self, roles: List[str]):
        assert self.role in roles
        if self.cont is None:
            ret = {role: hpc.Inaction() for role in roles}
        else:
            ret = self.cont.translate(roles)
        ret[self.role] = hpc.PrefixProcess(hpc.Assignment(self.var, self.expr), ret[self.role])
        return ret


class Communication(Fragment):
    def __init__(self, sender: str, receiver: str, channel: hpc.Channel,
                 var_list: List[str], expr_list: List[str], cont: Fragment):
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

        ret[self.sender] = hpc.PrefixProcess(
            hpc.Output(hpc.OutChannel(self.channel.name, self.channel.para), self.expr_list),
            ret[self.sender]
        )
        ret[self.receiver] = hpc.PrefixProcess(
            hpc.Input(hpc.InChannel(self.channel.name, self.channel.para), self.var_list),
            ret[self.receiver]
        )
        return ret



class Sensation(Fragment):
    def __init__(self, sender: str, receiver: str, var_x: str, var_v: str, cont: Fragment):
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
                hpc.Output(hpc.OutChannel(self.var_v), [self.expr]),
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
        print(f"role:{self.role},body:{self.body}")
        ready_set = get_ready_set(self.role, self.body)
        print(f"ready_set:{ready_set}")
        
        # Filter ready_set to keep only channels related to ODE variables.
        ode_vars = set(self.ode.v)
        filtered_ready_set = []
        for channel in ready_set:
            if channel.name in ode_vars:
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
        ret[self.role] = hpc.PrefixProcess(hpc.Wait(self.delay), ret[self.role])
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
        ret[self.role] = hpc.Conditional(self.cond, branch0, branch1)
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

    def __repr__(self):
        return f"Loop(index={self.index}, role={self.role}, cond={self.cond}, body={self.body})"

    def translate(self, roles):
        assert self.role in roles
        loop_index = self.index
        loop_channel = f"loop[{loop_index}]"

        # Translate loop.body into a process.
        loop_body_frag = self.body
        loop_body_proc = loop_body_frag.translate(roles)

        cont_proc = self.cont.translate(roles)

        ret = {}
        for role in roles:
            if role == self.role:
                cond = self.cond
                body_proc = loop_body_proc[role]

                # loop call
                loop_call = hpc.PrefixProcess(
                        hpc.Output(hpc.OutChannel(loop_channel), []),
                        hpc.Inaction()
                    )

                # then: loop_body + loop_call
                then_proc = sequence(body_proc, loop_call)

                else_proc = cont_proc[role]
                conditional = hpc.Conditional(condition=cond, branch0=then_proc, branch1=else_proc)
                loop_process = hpc.Loop(
                        channel=loop_channel,
                        formal_paras=[],
                        actual_paras=[],
                        process=conditional
                    )

                ret[role] = loop_process
            else:
                # loop_body + cont
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
            ret[role] = hpc.Parallel(proc0.get(role, hpc.Nil()), proc1.get(role, hpc.Nil()))
            ret[role] = hpc.Parallel(ret[role], cont_proc.get(role, hpc.Nil()))
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
                ready_set.append(hpc.OutChannel(f.var_v))
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

    # de-emphasize
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


class ESD:
    def __init__(self, frag: Fragment, roles: List[str]):
        assert len(roles) >= 2
        self.frag = frag
        self.roles = roles

    def translate(self):
        ret = self.frag.translate(self.roles)
        named_procs = [hpc.NamedProcess(role, ret[role]) for role in self.roles]
        return hpc.Parallel(named_procs)

def parse_conditional_expr(expr_str):
    cond_match = re.match(r"(\w+)\s*([<>=!]+)\s*(\w+)", expr_str)
    if cond_match:
        left, op, right = cond_match.groups()
        return hpc.Conditional(left=left, op=op, right=right)
    else:
        raise ValueError(f"Unparsable Conditional Expressions: {expr_str}")

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
        # === 1. loop ===
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

        # === 3. note delay(x) ===
        delay_match = re.match(r"note (left|right|over) of (\w+): delay\((\d+(?:\.\d+)?)\)", line)
        if delay_match:
            _, role, delay_value = delay_match.groups()
            delay_node = Delay(role=role, delay=float(delay_value), cont=Fragment())
            append_node(delay_node)
            continue


        # === 3b. note <<ode>> {...} ===
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

        # === 8. channels(...) ===
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


        # === 9. General communications ===
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
        if hasattr(current, "body") and current.body:
            print_fragment(current.body, indent + 1)
        current = getattr(current, "cont", None)


def main():
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()

    root = parse_example(example)
    print("=== Parse results ===")
    print_fragment(root)
    
    roles = ["Train", "LeftSector", "RightSector"]
    
    esd = ESD(root, roles)
    translated = esd.translate()

    print("\n=== Translation results ===")
    print(translated)

    with open("translated_output.txt", "w", encoding="utf-8") as f:
        f.write(str(translated))
        f.write("\n")

    print("Results have been saved to translated_output.txt")

if __name__ == "__main__":
    main()
