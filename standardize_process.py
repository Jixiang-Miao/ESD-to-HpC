# standardize_process.py
import re
from typing import Tuple, List, Set, Any, Dict, Optional
from hpc import *

_ident_re = re.compile(r"[a-zA-Z_]\w*")

def extract_vars_from_str(s: str) -> Set[str]:
    if s is None:
        return set()
    return {t for t in _ident_re.findall(str(s)) if not t.isdigit()}

def is_autonomous(ode: ODE) -> bool:
    used = set()
    for expr in getattr(ode, "e", []) or []:
        used |= extract_vars_from_str(expr)
    used |= extract_vars_from_str(getattr(ode, "bound", ""))
    return set(ode.v) >= used

def flatten_restrictions(obj: Any, seen=None) -> Tuple[Any, List[str]]:
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return obj, []
    seen.add(oid)

    if obj is None:
        return None, []

    if isinstance(obj, Restriction):
        inner, names = flatten_restrictions(obj.process, seen)
        return inner, list(names) + list(obj.names)

    if isinstance(obj, list):
        all_names = []
        new_list = []
        for item in obj:
            ni, n = flatten_restrictions(item, seen)
            new_list.append(ni)
            all_names += n
        return new_list, all_names

    if isinstance(obj, tuple):
        all_names = []
        new_tuple = []
        for item in obj:
            ni, n = flatten_restrictions(item, seen)
            new_tuple.append(ni)
            all_names += n
        return tuple(new_tuple), all_names

    if isinstance(obj, dict):
        all_names = []
        new_dict = {}
        for k, v in obj.items():
            nv, n = flatten_restrictions(v, seen)
            new_dict[k] = nv
            all_names += n
        return new_dict, all_names

    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes)):
        all_names = []
        for attr, val in list(vars(obj).items()):
            if isinstance(val, (Restriction, list, tuple, dict)) or hasattr(val, "__dict__"):
                new_val, n = flatten_restrictions(val, seen)
                try:
                    setattr(obj, attr, new_val)
                except Exception:
                    pass
                all_names += n
        return obj, all_names

    return obj, []

def extract_all_variables(process: Process) -> Set[str]:
    """Recursively extract all variable names from a process"""
    if process is None:
        return set()
    
    variables = set()
    
    if isinstance(process, str):
        variables.update(extract_vars_from_str(process))
    
    elif isinstance(process, Restriction):
        variables.update(process.names)
        variables.update(extract_all_variables(process.process))
    
    elif isinstance(process, PrefixProcess):
        if isinstance(process.prefix, Continuous):
            ode = process.prefix.ode
            variables.update(ode.v)
            variables.update(extract_vars_from_str(ode.bound))
            for e in ode.e0 + ode.e:
                variables.update(extract_vars_from_str(e))
            if hasattr(ode, 'ready_set') and ode.ready_set:
                variables.update(ode.ready_set)
            if hasattr(ode, 'final_v') and ode.final_v:
                variables.update(ode.final_v)
        variables.update(extract_all_variables(process.continuation))
    
    elif isinstance(process, Parallel):
        for p in process.processes:
            variables.update(extract_all_variables(p))
    
    elif isinstance(process, Sum):
        for b in process.branches:
            variables.update(extract_all_variables(b.prefix))
            variables.update(extract_all_variables(b.continuation))
    
    elif isinstance(process, (Loop, Replication, NamedProcess)):
        # handle different attribute names used across types
        if hasattr(process, 'process'):
            variables.update(extract_all_variables(process.process))
        if hasattr(process, 'body'):
            variables.update(extract_all_variables(process.body))
        if hasattr(process, 'definition'):
            variables.update(extract_all_variables(process.definition))
    
    elif isinstance(process, dict):
        for v in process.values():
            variables.update(extract_all_variables(v))
    
    elif isinstance(process, (list, tuple)):
        for item in process:
            variables.update(extract_all_variables(item))
    
    elif hasattr(process, "__dict__"):
        for val in vars(process).values():
            variables.update(extract_all_variables(val))
    
    return variables

def merge_non_autonomous_odes_in_parallel(p: Process) -> Process:
    if not isinstance(p, Parallel):
        return p

    to_merge = []
    others = []
    for part in p.processes:
        if isinstance(part, PrefixProcess) and isinstance(part.prefix, Continuous):
            if not is_autonomous(part.prefix.ode):
                to_merge.append(part)
            else:
                others.append(part)
        else:
            others.append(part)

    if len(to_merge) <= 1:
        return p

    all_v = []
    all_e0 = []
    all_e = []
    all_bounds = []
    all_ready = []
    all_final = []
    conts = []

    for pr in to_merge:
        ode = pr.prefix.ode
        all_v += ode.v
        all_e0 += ode.e0
        all_e += ode.e
        all_bounds.append(f"({ode.bound})")
        if getattr(ode, "ready_set", None):
            all_ready += ode.ready_set
        all_final += getattr(ode, "final_v", []) or []
        conts.append(pr.continuation)

    merged_ode = ODE(
        e0=all_e0,
        v=all_v,
        e=all_e,
        bound=" & ".join(all_bounds),
        ready_set=all_ready,
        final_v=all_final
    )

    merged_cont = Parallel(conts) if len(conts) > 1 else conts[0]
    merged_proc = PrefixProcess(Continuous(merged_ode), merged_cont)
    new_parts = others + [merged_proc]
    return Parallel(new_parts)

def merge_all_non_autonomous_odes(node, seen=None):
    if seen is None:
        seen = set()
    oid = id(node)
    if oid in seen:
        return node
    seen.add(oid)

    if node is None:
        return None
        
    if isinstance(node, Parallel):
        node.processes = [merge_all_non_autonomous_odes(p, seen) for p in node.processes]
        return merge_non_autonomous_odes_in_parallel(node)
        
    if isinstance(node, PrefixProcess):
        node.continuation = merge_all_non_autonomous_odes(node.continuation, seen)
        return node
        
    if isinstance(node, Sum):
        new_br = []
        for b in node.branches:
            b_cont = merge_all_non_autonomous_odes(b.continuation, seen)
            new_br.append(PrefixProcess(b.prefix, b_cont))
        node.branches = new_br
        return node
        
    if isinstance(node, Loop):
        node.process = merge_all_non_autonomous_odes(node.process, seen)
        return node
        
    if isinstance(node, Replication):
        node.process = merge_all_non_autonomous_odes(node.process, seen)
        return node
        
    if isinstance(node, NamedProcess):
        node.body = merge_all_non_autonomous_odes(node.body, seen)
        return node
        
    if hasattr(node, "__dict__") and not isinstance(node, (str, bytes)):
        for attr, val in list(vars(node).items()):
            if isinstance(val, (Process, list, tuple, dict)) or hasattr(val, "__dict__"):
                try:
                    setattr(node, attr, merge_all_non_autonomous_odes(val, seen))
                except Exception:
                    pass
        return node
        
    return node

def collect_component_definitions(top: Process) -> Dict[str, Process]:
    components = {}
    replication_count = 1

    def traverse(node, parent=None):
        nonlocal replication_count
        if node is None:
            return

        if isinstance(node, Parallel):
            new_processes = []
            for p in node.processes:
                traverse(p, parent=node)
                if isinstance(p, Replication):
                    rep_name = f"Replication {replication_count}"
                    components[rep_name] = p
                    replication_count += 1
                else:
                    new_processes.append(p)
            node.processes = new_processes
            return

        if isinstance(node, NamedProcess):
            if node.name not in components:
                components[node.name] = node.body
            traverse(node.body, parent=node)

        elif isinstance(node, PrefixProcess):
            traverse(node.continuation, parent=node)
        elif isinstance(node, Sum):
            for b in node.branches:
                traverse(b.continuation, parent=node)
        elif isinstance(node, (Loop, Replication)):
            if hasattr(node, 'process'):
                traverse(node.process, parent=node)
            if hasattr(node, 'body'):
                traverse(node.body, parent=node)
        elif hasattr(node, "__dict__") and not isinstance(node, (str, bytes)):
            for val in vars(node).values():
                if isinstance(val, (Process, list, tuple, dict)) or hasattr(val, "__dict__"):
                    traverse(val, parent=node)

    traverse(top)
    return components

def rename_variable_in_process(process: Process, old_name: str, new_name: str) -> Process:
    if process is None:
        return None
    
    if isinstance(process, str):
        return re.sub(rf'\b{old_name}\b', new_name, process)
    
    if isinstance(process, Restriction):
        new_names = [new_name if n == old_name else n for n in process.names]
        new_process = rename_variable_in_process(process.process, old_name, new_name)
        return Restriction(new_names, new_process)
    
    if isinstance(process, PrefixProcess):
        new_prefix = process.prefix
        if isinstance(process.prefix, Continuous):
            ode = process.prefix.ode
            new_v = [new_name if v == old_name else v for v in ode.v]
            new_e0 = [rename_variable_in_process(e, old_name, new_name) for e in ode.e0]
            new_e = [rename_variable_in_process(e, old_name, new_name) for e in ode.e]
            new_bound = rename_variable_in_process(ode.bound, old_name, new_name)
            new_ready = [new_name if r == old_name else r for r in (ode.ready_set or [])]
            new_final = [new_name if f == old_name else f for f in (ode.final_v or [])]
            new_ode = ODE(
                e0=new_e0,
                v=new_v,
                e=new_e,
                bound=new_bound,
                ready_set=new_ready,
                final_v=new_final
            )
            new_prefix = Continuous(new_ode)
        new_cont = rename_variable_in_process(process.continuation, old_name, new_name)
        return PrefixProcess(new_prefix, new_cont)
    
    if isinstance(process, Parallel):
        new_procs = [rename_variable_in_process(p, old_name, new_name) for p in process.processes]
        return Parallel(new_procs)
    
    if isinstance(process, Sum):
        new_branches = []
        for b in process.branches:
            new_prefix = b.prefix
            if hasattr(new_prefix, 'channel'):
                new_channel = rename_variable_in_process(new_prefix.channel, old_name, new_name)
                new_prefix.channel = new_channel
            if hasattr(new_prefix, 'expressions'):
                new_exprs = [rename_variable_in_process(e, old_name, new_name) 
                             for e in new_prefix.expressions]
                new_prefix.expressions = new_exprs
            new_cont = rename_variable_in_process(b.continuation, old_name, new_name)
            new_branches.append(PrefixProcess(new_prefix, new_cont))
        return Sum(new_branches)
    
    if isinstance(process, (Loop, Replication)):
        # 保持类型一致地重写内部
        if hasattr(process, 'process'):
            new_body = rename_variable_in_process(process.process, old_name, new_name)
            return type(process)(new_body)
        if hasattr(process, 'body'):
            new_body = rename_variable_in_process(process.body, old_name, new_name)
            return type(process)(new_body)
        return process
    
    if isinstance(process, NamedProcess):
        new_body = rename_variable_in_process(process.body, old_name, new_name)
        return NamedProcess(process.name, new_body)
    
    if isinstance(process, Block):
        new_defs = [rename_variable_in_process(d, old_name, new_name) for d in process.definitions]
        return Block(new_defs)
    
    if isinstance(process, dict):
        return {k: rename_variable_in_process(v, old_name, new_name) for k, v in process.items()}
    
    if isinstance(process, list):
        return [rename_variable_in_process(i, old_name, new_name) for i in process]
    
    if isinstance(process, tuple):
        return tuple(rename_variable_in_process(i, old_name, new_name) for i in process)
    
    if hasattr(process, "__dict__"):
        for attr, val in vars(process).items():
            try:
                setattr(process, attr, rename_variable_in_process(val, old_name, new_name))
            except Exception:
                pass
        return process
    
    return process

def resolve_name_conflicts(vars_to_restrict: List[str], components: Dict[str, Process]) -> Tuple[List[str], Dict[str, Process]]:
    """Scope-aware conflict resolution"""
    # Map var -> list of component names where it is *defined* (restriction variables)
    var_defined_in = {v: [] for v in vars_to_restrict}
    for comp_name, comp_body in components.items():
        comp_vars = extract_all_variables(comp_body)
        for v in vars_to_restrict:
            if v in comp_vars:
                var_defined_in[v].append(comp_name)

    renamed_components = {name: body for name, body in components.items()}
    final_vars = []
    for var, owners in var_defined_in.items():
        # debug prints retained
        print({var})
        print(list(owners))  # Convert to list to print actual contents
        if len(owners) <= 1:
            final_vars.append(var)  # no conflict; keep as-is
            continue
        # Conflict — rename in all but first owner
        final_vars.append(var)  # keep original for first owner
        for i, comp_name in enumerate(owners[1:], start=1):
            if comp_name.startswith("Memory"):
                continue  # skip memory components
            new_name = var + "'" * i
            print(f"comp_name:{comp_name}")
            final_vars.append(new_name)
            renamed_components[comp_name] = rename_variable_in_process(renamed_components[comp_name], var, new_name)
    print(f"final_vars, renamed_components:{final_vars, list(renamed_components)}")
    return final_vars, renamed_components

def to_standard_form(top: Process) -> Process:
    components = collect_component_definitions(top)

    flattened_top, top_names = flatten_restrictions(top)

    processed_top = merge_all_non_autonomous_odes(flattened_top)

    all_names = list(top_names)
    for comp_name, comp_body in components.items():
        _, comp_names = flatten_restrictions(comp_body)
        if comp_names:
            all_names.extend(comp_names)

    unique_vars = list(dict.fromkeys(all_names))

    final_vars, final_components = resolve_name_conflicts(unique_vars, components)

    component_definitions = []
    defined_names = set()
    for name, body in final_components.items():
        if name not in defined_names:
            component_definitions.append(NamedProcess(name=name, body=body))
            defined_names.add(name)

    class SimpleParallel(Parallel):
        def __str__(self):
            return " || ".join(p.name if hasattr(p, 'name') else str(p) for p in self.processes)
    system_parallel = SimpleParallel([
        NamedProcess(name=name, body=None)
        for name in final_components.keys()
    ])

    current_body = system_parallel
    for var in reversed(final_vars):
        current_body = Restriction([var], current_body)

    system_definition = NamedProcess(name="System", body=current_body)

    return Block(definitions=component_definitions + [system_definition])

class Block(Process):
    def __init__(self, definitions: List[Process]):
        self.definitions = definitions
    def __repr__(self):
        return "\n".join(str(d) for d in self.definitions)

def define_component(name: str, process: Process) -> NamedProcess:
    return NamedProcess(name=name, body=process)
