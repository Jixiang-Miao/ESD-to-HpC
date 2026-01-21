# simulator.py
import math
import re
import copy
import unicodedata
from typing import List, Dict, Tuple, Any, Optional, Set, Union
import hpc
from esd import *
from hpc import *
import numpy as np
from scipy.integrate import solve_ivp

_ident_re = re.compile(r"[a-zA-Z_][\w']*")

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
    var_defined_in = {v: [] for v in vars_to_restrict}
    for comp_name, comp_body in components.items():
        comp_vars = extract_all_variables(comp_body)
        for v in vars_to_restrict:
            if v in comp_vars:
                var_defined_in[v].append(comp_name)

    renamed_components = {name: body for name, body in components.items()}
    final_vars = []
    for var, owners in var_defined_in.items():
        print({var})
        print(list(owners))
        if len(owners) <= 1:
            final_vars.append(var)
            continue
        final_vars.append(var)
        for i, comp_name in enumerate(owners[1:], start=1):
            if comp_name.startswith("Memory"):
                continue
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

class ChannelPayload:
    def __init__(self, name: str, args: Tuple[Any, ...]):
        self.name = name
        self.args = args
    def __repr__(self):
        try:
            return f"{self.name}({', '.join(map(str, self.args))})"
        except Exception:
            return f"{self.name}(<args>)"
    def __str__(self):
        return self.__repr__()

# ---------- global env ----------
EVENTS_FILE = "simulation_events.txt"
TRACE_FILE = "trace.txt"
STEP_FILE = "steps.txt"
DEFAULT_INIT_STATE = {}
MAX_CONTINUOUS_STEP = 10.0
CONTINUOUS_PRECISION = 1e-6
VERBOSE_STEPS = 1
TERMINATE_ON_PHYSICAL_BOUNDARY = False

global_env: Dict[str, float] = {}
alias_map: Dict[str, str] = {}
hidden_vars: Set[str] = set()
events_buffer: List[str] = []
trace_buffer: List[str] = []
step_buffer: List[str] = []
LOG_FLUSH_THRESHOLD = 50
function_impls: Dict[str, Any] = {}

def flush_logs():
    """Flush the buffered logs to disk"""
    global events_buffer, trace_buffer, step_buffer
    try:
        if events_buffer:
            with open(EVENTS_FILE, "a", encoding="utf-8") as f:
                f.write("".join(e + "\n" for e in events_buffer))
            events_buffer = []
        if trace_buffer:
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write("".join(t + "\n" for t in trace_buffer))
            trace_buffer = []
        if step_buffer:
            with open(STEP_FILE, "a", encoding="utf-8") as f:
                f.write("".join(s + "\n" for s in step_buffer))
            step_buffer = []
    except Exception as e:
        print(f"[Log Error] flush_logs failed: {e}")

def set_initial_state(init_state: dict):
    """Set the initial state"""
    global global_env
    global_env = copy.deepcopy(DEFAULT_INIT_STATE)
    global_env.update(init_state)

def add_hidden_vars(vars: List[str]):
    """Add hidden variables"""
    global hidden_vars
    hidden_vars.update(vars)

# ---------- Core functions for canonical form processing (fixing Sum process loss issue) ----------
def classify_process_type(proc: hpc.Process) -> str:
    """Classify process as M/C/R type"""
    if proc is None:
        return "other"

    if isinstance(proc, hpc.Sum) or isinstance(proc, hpc.Inaction):
        return "M"
    if isinstance(proc, hpc.Replication):
        return "R"
    if isinstance(proc, hpc.PrefixProcess) and isinstance(getattr(proc, 'prefix', None), hpc.Continuous):
        return "C"

    if isinstance(proc, hpc.PrefixProcess):
        pref = getattr(proc, 'prefix', None)
        if isinstance(pref, (hpc.Guard, hpc.Input, hpc.Output)):
            return "M"
        return classify_process_type(getattr(proc, 'continuation', None))

    if isinstance(proc, hpc.Parallel):
        has_m = False
        has_c = False
        has_r = False
        for sub in getattr(proc, 'processes', []):
            t = classify_process_type(sub)
            if t == "M":
                has_m = True
            elif t == "C":
                has_c = True
            elif t == "R":
                has_r = True

        if has_m:
            return "M"
        if has_c:
            return "C"
        if has_r:
            return "R"
        return "other"

    if isinstance(proc, hpc.Restriction):
        return classify_process_type(getattr(proc, 'process', None))
    if isinstance(proc, hpc.Replication):
        return classify_process_type(getattr(proc, 'process', None))
    if isinstance(proc, hpc.NamedProcess):
        body = getattr(proc, 'body', None) or getattr(proc, 'process', None)
        return classify_process_type(body)

    return "other"

def to_strict_canonical_form_preserve_sums(proc: hpc.Process) -> hpc.Process:
    """Convert to strict canonical form but specially protect Sum processes"""
    if proc is None:
        return hpc.Inaction()
    
    if isinstance(proc, hpc.Sum):
        log_event("Protecting Sum process during canonical form conversion")
        branches = get_sum_branches(proc)
        new_branches = []
        for branch in branches:
            new_branch = to_strict_canonical_form_preserve_sums(branch)
            if not isinstance(new_branch, hpc.Inaction):
                new_branches.append(new_branch)
        
        if not new_branches:
            return hpc.Inaction()
        elif len(new_branches) == 1:
            return new_branches[0]
        else:
            return hpc.Sum(branches=new_branches)
    
    if isinstance(proc, hpc.Inaction):
        return proc
    
    if isinstance(proc, hpc.Parallel):
        flat_procs = []
        for p in proc.processes:
            canonical_p = to_strict_canonical_form_preserve_sums(p)
            if isinstance(canonical_p, hpc.Parallel):
                flat_procs.extend(canonical_p.processes)
            elif not isinstance(canonical_p, hpc.Inaction):
                flat_procs.append(canonical_p)
        
        flat_procs = [p for p in flat_procs if not isinstance(p, hpc.Inaction)]
        
        if not flat_procs:
            return hpc.Inaction()
        elif len(flat_procs) == 1:
            return flat_procs[0]
        else:
            return rebuild_canonical_form(flat_procs)
    
    if isinstance(proc, hpc.PrefixProcess):
        new_continuation = to_strict_canonical_form_preserve_sums(proc.continuation)
        return hpc.PrefixProcess(proc.prefix, new_continuation)
    
    if isinstance(proc, hpc.Replication):
        new_process = to_strict_canonical_form_preserve_sums(proc.process)
        return hpc.Replication(new_process)
    
    if isinstance(proc, hpc.Restriction):
        new_process = to_strict_canonical_form_preserve_sums(proc.process)
        
        if isinstance(new_process, hpc.Inaction):
            return hpc.Inaction()
        
        if isinstance(new_process, hpc.Parallel):
            free_vars = get_free_vars(new_process)
            target_vars = set(proc.names)
            
            restricted_procs = []
            unrestricted_procs = []
            
            for p in new_process.processes:
                p_free_vars = get_free_vars(p)
                if target_vars & p_free_vars:
                    restricted_procs.append(hpc.Restriction(proc.names, p))
                else:
                    unrestricted_procs.append(p)
            
            all_procs = unrestricted_procs + restricted_procs
            if len(all_procs) == 1:
                return all_procs[0]
            else:
                return hpc.Parallel(all_procs)
        
        return hpc.Restriction(proc.names, new_process)
    
    return proc

def rebuild_canonical_form(procs: List[hpc.Process]) -> hpc.Parallel:
    """Rebuild the process list into a canonical parallel composition (M || C || R)"""
    m_procs = []
    c_procs = []
    r_procs = []
    other_procs = []
    
    for proc in procs:
        proc_type = classify_process_type(proc)
        
        if proc_type == "M":
            m_procs.append(proc)
        elif proc_type == "C":
            c_procs.append(proc)
        elif proc_type == "R":
            r_procs.append(proc)
        elif proc_type != "inaction":
            other_procs.append(proc)
    
    all_procs = m_procs + c_procs + r_procs + other_procs
    
    if not all_procs:
        return hpc.Parallel([hpc.Inaction()])
    
    return hpc.Parallel(all_procs)

def validate_canonical_form(proc: hpc.Process) -> Tuple[bool, str]:
    """Validate if the process conforms to the canonical form"""
    if proc is None:
        return False, "Process is None"
    
    if isinstance(proc, hpc.Inaction):
        return True, "Empty process is a valid canonical form"
    
    if not isinstance(proc, hpc.Parallel):
        return False, f"Top-level process should be of type Parallel, but got {type(proc).__name__}"
    
    m_count = 0
    c_count = 0
    r_count = 0
    other_count = 0
    
    for sub_proc in proc.processes:
        proc_type = classify_process_type(sub_proc)
        if proc_type == "M":
            m_count += 1
        elif proc_type == "C":
            c_count += 1
        elif proc_type == "R":
            r_count += 1
        else:
            other_count += 1
    
    if m_count == 0:
        return False, "Canonical form is missing M type processes (Sum processes)"
    
    found_types = []
    current_section = "M"
    
    for sub_proc in proc.processes:
        proc_type = classify_process_type(sub_proc)
        
        if proc_type == "M":
            if current_section not in ["M"]:
                return False, f"Process type order error: M type process appears after {current_section} section"
        elif proc_type == "C":
            if current_section == "M":
                current_section = "C"
            elif current_section not in ["C"]:
                return False, f"Process type order error: C type process appears after {current_section} section"
        elif proc_type == "R":
            if current_section in ["M", "C"]:
                current_section = "R"
            elif current_section not in ["R"]:
                return False, f"Process type order error: R type process appears after {current_section} section"
    
    return True, f"Canonical form validation passed: M={m_count}, C={c_count}, R={r_count}, Others={other_count}"

def count_m_processes(proc: hpc.Process) -> int:
    """Count the number of M type processes"""
    if isinstance(proc, hpc.Parallel):
        count = 0
        for sub_proc in proc.processes:
            if classify_process_type(sub_proc) == "M":
                count += 1
        return count
    else:
        return 1 if classify_process_type(proc) == "M" else 0

def repair_missing_sums(proc: hpc.Process, defs: Dict[str, hpc.Process]) -> hpc.Process:
    """Repair missing Sum processes"""    
    if isinstance(proc, hpc.Parallel):
        new_procs = list(proc.processes)
        
        has_m = any(classify_process_type(p) == "M" for p in new_procs)
        
        if not has_m:
            for name, body in defs.items():
                if has_sum_structure(body):
                        sum_proc = extract_sum_from_process(body)
                        if sum_proc:
                            new_procs.insert(0, sum_proc)
                            break
            if not any(classify_process_type(p) == "M" for p in new_procs):
                base_sum = create_base_sum_process()
                new_procs.insert(0, base_sum)
        
        return hpc.Parallel(new_procs)
    
    return proc

def has_sum_structure(proc: hpc.Process) -> bool:
    """Check if the process contains a Sum structure"""
    if isinstance(proc, hpc.Sum):
        return True
    elif isinstance(proc, hpc.PrefixProcess):
        return has_sum_structure(proc.continuation)
    elif isinstance(proc, hpc.Parallel):
        return any(has_sum_structure(p) for p in proc.processes)
    elif isinstance(proc, hpc.Replication):
        return has_sum_structure(proc.process)
    elif isinstance(proc, hpc.Restriction):
        return has_sum_structure(proc.process)
    elif isinstance(proc, hpc.NamedProcess):
        return has_sum_structure(proc.body)
    return False

def extract_sum_from_process(proc: hpc.Process) -> Optional[hpc.Sum]:
    """Extract Sum structure from process"""
    if isinstance(proc, hpc.Sum):
        return copy.deepcopy(proc)
    elif isinstance(proc, hpc.PrefixProcess):
        return extract_sum_from_process(proc.continuation)
    elif isinstance(proc, hpc.Parallel):
        for p in proc.processes:
            result = extract_sum_from_process(p)
            if result:
                return result
    elif isinstance(proc, hpc.Replication):
        return extract_sum_from_process(proc.process)
    elif isinstance(proc, hpc.Restriction):
        return extract_sum_from_process(proc.process)
    elif isinstance(proc, hpc.NamedProcess):
        return extract_sum_from_process(proc.body)
    return None

def create_base_sum_process() -> hpc.Sum:
    """Create a base Sum process for initializing communication"""
    input_branch = hpc.PrefixProcess(
        prefix=hpc.Input(channel="p", var_list=["p0"]),
        continuation=hpc.Inaction()
    )
    
    return hpc.Sum(branches=[input_branch])

# ---------- Helper Functions ----------
def normalize_channel_name(raw: str) -> str:
    """Normalize channel name, preserving indexed channel names (e.g., loop[1])"""
    if raw is None:
        return ""
    
    s = str(raw).strip()
    normalized = unicodedata.normalize("NFD", s)
    base_name = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    base_name = base_name.replace(" ", "")
    
    if '[' in base_name and ']' in base_name:
        pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]$'
        match = re.match(pattern, base_name)
        if match:
            base = match.group(1)
            index = match.group(2)
            return f"{base}[{index}]"
        else:
            pattern = r'([a-zA-Z0-9_]+)\[(\d+)\]'
            match = re.search(pattern, base_name)
            if match:
                base = match.group(1)
                index = match.group(2)
                return f"{base}[{index}]"
    
    pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)'
    match = re.search(pattern, base_name)
    
    return match.group(1) if match else base_name

def channel_to_str(channel: Any) -> str:
    """Extract the real channel name"""
    if channel is None:
        return ""
    if hasattr(channel, 'name'):
        name_str = str(channel.name).strip()
    else:
        name_str = str(channel).strip()
    pattern = r'^(allocation|store|load)\((.+?)\)$'
    match = re.match(pattern, name_str, re.IGNORECASE)
    return match.group(2).strip() if match else name_str

def get_sum_branches(sum_proc: Union[hpc.Sum, hpc.PrefixProcess]) -> List[hpc.PrefixProcess]:
    """Get all branches of a Sum process"""
    branches = []
    if isinstance(sum_proc, hpc.Sum):
        for sub_proc in sum_proc.branches:
            branches.extend(get_sum_branches(sub_proc))
    elif isinstance(sum_proc, hpc.PrefixProcess):
        branches.append(sum_proc)
    return branches

def get_free_vars(proc: hpc.Process) -> Set[str]:
    """Calculate the set of free variables in a process"""
    if isinstance(proc, hpc.Inaction):
        return set()
    
    elif isinstance(proc, hpc.PrefixProcess):
        free = get_free_vars(proc.continuation)
        if isinstance(proc.prefix, hpc.Input):
            free -= set(proc.prefix.var_list)
        elif isinstance(proc.prefix, hpc.Guard):
            free.update(identifiers_in_expr(proc.prefix.condition))
        elif isinstance(proc.prefix, hpc.Output):
            for expr in proc.prefix.expr_list:
                free.update(identifiers_in_expr(str(expr)))
        return free
    
    elif isinstance(proc, hpc.Sum):
        free = set()
        for b in get_sum_branches(proc):
            free.update(get_free_vars(b))
        return free
    
    elif isinstance(proc, hpc.Parallel):
        free = set()
        for p in proc.processes:
            free.update(get_free_vars(p))
        return free
    
    elif isinstance(proc, hpc.Replication):
        return get_free_vars(proc.process)
    
    elif isinstance(proc, hpc.Restriction):
        free = get_free_vars(proc.process)
        free -= set(proc.names)
        return free
    
    elif isinstance(proc, hpc.Continuous):
        free = set()
        for e in proc.ode.e0 + proc.ode.e:
            free.update(identifiers_in_expr(str(e)))
        free.update(identifiers_in_expr(str(proc.ode.bound)))
        free -= set(proc.ode.v)
        return free
    
    return set()

def replace_first_continuous(component, replacement):
    """
    Recursively search for the first PrefixProcess with prefix Continuous in component,
    replace that PrefixProcess node with replacement (Process), and return the new component.
    Only replace the first encountered Continuous.
    """
    replaced = {"done": False}

    def _rec(node):
        if node is None:
            return node
        
        if isinstance(node, hpc.PrefixProcess) and isinstance(getattr(node, "prefix", None), hpc.Continuous):
            if not replaced["done"]:
                replaced["done"] = True
                return replacement
            return node
        
        if isinstance(node, hpc.PrefixProcess):
            new_cont = _rec(node.continuation)
            if new_cont is node.continuation:
                return node
            return hpc.PrefixProcess(node.prefix, new_cont)
        
        if isinstance(node, hpc.Sum):
            new_branches = []
            changed = False
            for br in node.branches:
                nb_cont = _rec(br.continuation)
                if nb_cont is not br.continuation:
                    changed = True
                    new_branches.append(hpc.PrefixProcess(br.prefix, nb_cont))
                else:
                    new_branches.append(br)
                if replaced["done"]:
                    new_branches.extend(node.branches[len(new_branches):])
                    break
            if not changed:
                return node
            return hpc.Sum(new_branches)
        if isinstance(node, hpc.Replication):
            new_inner = _rec(node.process)
            if new_inner is node.process:
                return node
            return hpc.Replication(new_inner)
        if isinstance(node, hpc.Restriction):
            new_proc = _rec(node.process)
            if new_proc is node.process:
                return node
            return hpc.Restriction(node.names, new_proc)
        if isinstance(node, hpc.NamedProcess):
            body = getattr(node, "body", None) or getattr(node, "process", None)
            new_body = _rec(body)
            if new_body is body:
                return node
            return hpc.NamedProcess(node.name, new_body)
        if isinstance(node, hpc.Parallel):
            new_procs = []
            changed = False
            for sub in node.processes:
                new_sub = _rec(sub)
                new_procs.append(new_sub)
                if new_sub is not sub:
                    changed = True
                if replaced["done"]:
                    new_procs.extend(node.processes[len(new_procs):])
                    break
            if not changed:
                return node
            return hpc.Parallel(new_procs)
        return node

    return _rec(component)

def expand_to_basic_types_with_sums(proc: hpc.Process, defs: Dict[str, hpc.Process], visited: Set[str] = None) -> hpc.Process:
    """Expand named process references to basic types, preserving Sum structures for further processing.
    This is a conservative recursive expansion implementation: when encountering NamedProcess, it attempts to find the definition in defs and expand it,
    recursively expands child structures for Parallel/Sum/Prefix/Replication/Restriction.
    Uses the visited set to avoid infinite recursion caused by recursive expansion.
    """
    if visited is None:
        visited = set()

    if proc is None:
        return hpc.Inaction()

    if isinstance(proc, hpc.NamedProcess):
        name = getattr(proc, 'name', None)
        if not name:
            body = getattr(proc, 'body', None) or getattr(proc, 'process', None)
            return expand_to_basic_types_with_sums(body, defs, visited)

        if name in visited:
            return copy.deepcopy(defs.get(name, proc))

        if name in defs:
            visited.add(name)
            body = copy.deepcopy(defs[name])
            expanded = expand_to_basic_types_with_sums(body, defs, visited)
            visited.remove(name)
            return expanded
        else:
            body = getattr(proc, 'body', None) or getattr(proc, 'process', None)
            return expand_to_basic_types_with_sums(body, defs, visited)

    if isinstance(proc, hpc.PrefixProcess):
        new_cont = expand_to_basic_types_with_sums(proc.continuation, defs, visited)
        return hpc.PrefixProcess(proc.prefix, new_cont)

    if isinstance(proc, hpc.Parallel):
        new_list = [expand_to_basic_types_with_sums(p, defs, visited) for p in proc.processes]
        return hpc.Parallel(new_list)

    if isinstance(proc, hpc.Sum):
        branches = get_sum_branches(proc)
        new_branches = [expand_to_basic_types_with_sums(b, defs, visited) for b in branches]
        return _reconstruct_sum(new_branches)

    if isinstance(proc, hpc.Replication):
        inner = getattr(proc, 'process', None) or getattr(proc, 'body', None)
        new_inner = expand_to_basic_types_with_sums(inner, defs, visited)
        return hpc.Replication(new_inner)

    if isinstance(proc, hpc.Restriction):
        new_inner = expand_to_basic_types_with_sums(proc.process, defs, visited)
        return hpc.Restriction(proc.names, new_inner)

    return copy.deepcopy(proc)

def safe_eval_expr(expr: str, env: Dict[str, float], par: Optional[hpc.Parallel] = None) -> Tuple[Any, Set[str]]:
    """Safely evaluate an expression"""
    global _SAFE_EVAL_DEPTH
    try:
        _SAFE_EVAL_DEPTH
    except NameError:
        _SAFE_EVAL_DEPTH = 0

    if _SAFE_EVAL_DEPTH > 60:
        return None, set()

    if expr is None or str(expr).strip() == "":
        return None, set()

    s = str(expr).strip()
    s2 = s.replace("^", "**").replace("¬", " not ").replace("&&", " and ").replace("||", " or ")

    if len(s2) > 1000:
        try:
            return float(s2), set()
        except Exception:
            try:
                ids_quick = identifiers_in_expr(s2)
            except Exception:
                ids_quick = set()
            return None, ids_quick

    ids = identifiers_in_expr(s2)
    math_funcs = {"abs", "sqrt", "sin", "cos", "exp", "log", "tan", "floor", "ceil"}
    math_consts = {"pi", "e"}

    missing_vars = set(v for v in ids if v not in env and v not in math_funcs and v not in math_consts and v != 't')
    if missing_vars:
        return None, missing_vars

    eval_globals = {"__builtins__": None}
    for fn in ("abs", "sqrt", "sin", "cos", "exp", "log", "tan", "floor", "ceil"):
        if hasattr(math, fn):
            eval_globals[fn] = getattr(math, fn)
    eval_globals['pi'] = math.pi
    eval_globals['e'] = math.e

    safe_map = {}
    for ident in ids:
        safe_name = ident.replace("'", "_prime")
        safe_name = re.sub(r'[^0-9a-zA-Z_]', '_', safe_name)
        if re.match(r'^[0-9]', safe_name):
            safe_name = "_" + safe_name
        safe_map[ident] = safe_name

    eval_locals: Dict[str, Any] = {}
    for ident in ids:
        _SAFE_EVAL_DEPTH += 1
        try:
            if ident in env:
                val = env.get(ident)
                if isinstance(val, (int, float, bool)):
                    eval_locals[safe_map[ident]] = val
                    continue
                elif callable(val):
                    eval_locals[safe_map[ident]] = val
                    continue
                else:
                    try:
                        if isinstance(val, ChannelPayload) and len(val.args) > 0:
                            candidate = val.args[0]
                            try:
                                eval_locals[safe_map[ident]] = float(candidate)
                                continue
                            except Exception:
                                v, missing = safe_eval_expr(str(candidate), env)
                                if not missing and isinstance(v, (int, float, bool)):
                                    eval_locals[safe_map[ident]] = v
                                    continue
                        _SAFE_EVAL_DEPTH -= 1
                        return None, {ident}
                    except Exception:
                        _SAFE_EVAL_DEPTH -= 1
                        return None, {ident}
            else:
                _SAFE_EVAL_DEPTH -= 1
                return None, {ident}
        finally:
            _SAFE_EVAL_DEPTH -= 1

    s_safe = s2
    for ident in sorted(ids, key=lambda x: -len(x)):
        s_safe = re.sub(rf'(?<!\w){re.escape(ident)}(?!\w)', safe_map[ident], s_safe)
    try:
        code_obj = compile(s_safe, '<safe_eval>', 'eval')
        val = eval(code_obj, eval_globals, eval_locals)
        return val, set()
    except KeyboardInterrupt:
        try:
            ids_quick = identifiers_in_expr(s2)
        except Exception:
            ids_quick = set()
        return None, ids_quick
    except Exception:
        try:
            return float(s2), set()
        except Exception:
            if s2.lower() in ("true", "false"):
                return s2.lower() == "true", set()
            return None, ids

def identifiers_in_expr(s: str) -> Set[str]:
    """Extract identifiers from an expression"""
    if s is None:
        return set()
    text = str(s)
    if len(text) > 5000:
        try:
            return set()
        except Exception:
            return set()

    try:
        tokens = re.findall(r"(?<!\w)[A-Za-z_][A-Za-z0-9_']*(?!\w)", text)
        if len(tokens) > 500:
            tokens = tokens[:500]
    except KeyboardInterrupt:
        return set()
    except Exception:
        return set()

    keywords = {"and", "or", "not", "True", "False", "None", "pi", "e"}
    import keyword as _py_keyword

    ids = []
    for t in tokens:
        try:
            if not t:
                continue
            if t in keywords:
                continue
            if _py_keyword.iskeyword(t):
                continue

            try:
                _ = float(t)
                continue
            except Exception:
                pass

            if len(t) > 200:
                continue

            if t not in ids:
                ids.append(t)
        except KeyboardInterrupt:
            return set()
        except Exception:
            continue

    return set(ids)

def try_eval_output_expr(expr: Any, par: Optional[hpc.Parallel] = None) -> Tuple[Any, Set[str]]:
    """Enhanced expression evaluation function - fixes symbol resolution issues"""
    try:
        v, missing = safe_eval_expr(str(expr), global_env, par)
        if missing == set() and v is not None:
            return v, set()
    except Exception:
        v, missing = None, set()

    s = str(expr).strip()
    m = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", s)
    if not m:
        return None, missing if missing else set()

    fname = m.group(1)
    args_str = m.group(2).strip()
    
    if args_str == "":
        args = []
    else:
        args = [a.strip() for a in args_str.split(',')]

    evaluated_args = []
    missing_total = set()
    local_resolved = {}
    
    for a in args:
        val, miss = safe_eval_expr(str(a), global_env, par)
        
        if miss:
            resolved_val = resolve_missing_variable(a, par, miss)
            if resolved_val is not None:
                val = resolved_val
                local_resolved.update({var: resolved_val for var in miss})
                miss = set()
        
        if miss:
            missing_total.update(miss)
        
        evaluated_args.append(val)

    if missing_total and par is not None:
        newly_resolved = resolve_variables_from_process(par, missing_total)
        if newly_resolved:
            temp_env = dict(global_env)
            temp_env.update(newly_resolved)
            evaluated_args = []
            missing_total = set()
            
            for a in args:
                val, miss = safe_eval_expr(str(a), temp_env, par)
                if miss:
                    missing_total.update(miss)
                evaluated_args.append(val)

    if not missing_total:
        if fname in function_impls and callable(function_impls[fname]):
            try:
                result = function_impls[fname](*evaluated_args)
                return result, set()
            except Exception as e:
                log_event(f"Function call failed: {fname} Error: {e}")

        if fname in global_env and callable(global_env[fname]):
            try:
                result = global_env[fname](*evaluated_args)
                return result, set()
            except Exception as e:
                log_event(f"Global function call failed: {fname} Error: {e}")

    if missing_total:
        return ChannelPayload(fname, tuple(args)), missing_total
    
    return None, missing_total

def resolve_missing_variable(expr: str, par: hpc.Parallel, missing_vars: Set[str]) -> Optional[float]:
    """Resolve missing variables - infer values from process communication"""
    if len(missing_vars) != 1:
        return None
    
    missing_var = next(iter(missing_vars))
    
    try:
        return float(expr)
    except (ValueError, TypeError):
        pass
    
    return extract_constant_from_sum(par, missing_var)

def resolve_variables_from_process(par: hpc.Parallel, missing_vars: Set[str]) -> Dict[str, float]:
    """Resolve multiple missing variables from process structure"""
    resolved = {}
    
    for var in missing_vars:
        val = extract_constant_from_sum(par, var)
        if val is not None:
            resolved[var] = val
            continue
        val = infer_value_from_communication(par, var)
        if val is not None:
            resolved[var] = val
    
    return resolved

def infer_value_from_communication(par: hpc.Parallel, var_name: str) -> Optional[float]:
    """Infer variable value from communication channels"""
    for proc in getattr(par, 'processes', []):
        if isinstance(proc, hpc.PrefixProcess) and isinstance(proc.prefix, hpc.Output):
            channel = channel_to_str(proc.prefix.channel)
            if var_name in channel:
                exprs = getattr(proc.prefix, 'expr_list', [])
                if exprs:
                    try:
                        val, missing = safe_eval_expr(str(exprs[0]), global_env, par)
                        if not missing:
                            return float(val)
                    except Exception:
                        pass
    return None

# ---------- process preparation functions----------
def prepare_system_from_block(block):
    """Prepare a system process in canonical form from a Block (fixing missing Sum processes)"""
    defs = collect_named_definitions(block)
    
    system_proc = find_system_process(block)
    if system_proc is None:
        raise RuntimeError("No System process found in block")
    
    expanded_proc = expand_to_basic_types_with_sums(system_proc, defs)
    
    unwrapped_proc = unwrap_restriction(expanded_proc)
    
    canonical_proc = to_strict_canonical_form_preserve_sums(unwrapped_proc)
    
    is_valid, message = validate_canonical_form(canonical_proc)
    
    if not is_valid or count_m_processes(canonical_proc) == 0:
        canonical_proc = repair_missing_sums(canonical_proc, defs)
        is_valid, message = validate_canonical_form(canonical_proc)
    
    try:
        initialize_allocations(canonical_proc, defs)
    except Exception as e:
        log_event(f"Warning: Error initializing allocation constants: {e}")
    
    try:
        initialize_placeholders(canonical_proc, defs)
    except Exception as e:
        log_event(f"Warning: Error initializing placeholder symbols: {e}")
    
    return canonical_proc, defs

def collect_named_definitions(block: Block) -> Dict[str, hpc.Process]:
    """Collect named definitions"""
    defs = {}
    if block is None:
        return defs

    definitions = getattr(block, "definitions", None) or []
    
    for i, d in enumerate(definitions):
        try:
            if isinstance(d, hpc.NamedProcess):
                name = getattr(d, "name", None)
                body = getattr(d, "body", None) or getattr(d, "process", None)
                if name and body:
                    defs[name] = copy.deepcopy(body)
                    log_event(f"Collected named definition[{i}]: {name}")
        except Exception as e:
            log_event(f"WARNING: Failed to collect definition[{i}]: {e}")
            continue
    
    return defs

def find_system_process(block: Block):
    """Find the System process"""
    if block is None:
        return None
        
    definitions = getattr(block, "definitions", None) or []
    
    for d in definitions:
        try:
            if isinstance(d, hpc.NamedProcess) and getattr(d, "name", None) == "System":
                return d
        except Exception:
            continue
    
    if definitions:
        return definitions[0]
    
    return None

def unwrap_restriction(proc: hpc.Process) -> hpc.Process:
    """Remove restriction wrappers"""
    if proc is None:
        return hpc.Inaction()
    
    if isinstance(proc, hpc.Restriction):
        log_event(f"Removing restriction: {proc.names}")
        inner = unwrap_restriction(proc.process)
        return inner
    
    return proc

def initialize_allocations(par: hpc.Parallel, defs: Dict[str, hpc.Process] = None):
    """Initialize allocation constants"""
    try:
        for proc in getattr(par, 'processes', []):
            candidates = []
            if isinstance(proc, hpc.PrefixProcess):
                candidates.append(proc)
            elif isinstance(proc, hpc.Sum):
                candidates.extend(get_sum_branches(proc))
            elif isinstance(proc, hpc.Replication):
                inner = getattr(proc, 'process', None) or getattr(proc, 'body', None)
                if isinstance(inner, hpc.PrefixProcess):
                    candidates.append(inner)
                elif isinstance(inner, hpc.Sum):
                    candidates.extend(get_sum_branches(inner))

            for cand in candidates:
                try:
                    if not (isinstance(cand, hpc.PrefixProcess) and isinstance(cand.prefix, hpc.Output)):
                        continue
                    out_pref = cand.prefix
                    ch_raw = channel_to_str(out_pref.channel)
                    base = normalize_channel_name(ch_raw)

                    m = re.match(r'^allocation\((.+)\)$', ch_raw)
                    if m:
                        name = m.group(1)
                    else:
                        name = base

                    exprs = getattr(out_pref, 'expr_list', None) or getattr(out_pref, 'exprs', None) or []
                    if exprs:
                        expr = exprs[0]
                        val, missing = safe_eval_expr(str(expr), global_env, par)
                        if missing == set() and val is not None:
                            try:
                                global_env[name] = float(val)
                                log_event(f"Initialized allocation constant: {name}={val}")
                            except Exception:
                                pass
                except Exception:
                    continue
    except Exception as e:
        log_event(f"Warning: Failed to initialize allocation constants: {e}")

def initialize_placeholders(par: hpc.Parallel, defs: Dict[str, hpc.Process] = None):
    """Initialize placeholders"""
    global global_env
    try:
        if 'f' not in global_env:
            def f_constructor(*args):
                resolved = []
                for a in args:
                    if isinstance(a, (int, float)):
                        resolved.append(float(a))
                        continue
                    if isinstance(a, ChannelPayload):
                        try:
                            val = a.args[0]
                            v, missing = safe_eval_expr(str(val), global_env)
                            if not missing and v is not None:
                                resolved.append(float(v))
                                continue
                        except Exception:
                            pass
                        return ChannelPayload('f', tuple(args))
                    try:
                        v, missing = safe_eval_expr(str(a), global_env)
                        if missing == set() and v is not None:
                            resolved.append(float(v))
                            continue
                        else:
                            return ChannelPayload('f', tuple(args))
                    except Exception:
                        return ChannelPayload('f', tuple(args))

                if 'f' in function_impls and callable(function_impls['f']):
                    try:
                        return function_impls['f'](*resolved)
                    except Exception as e:
                        return 0.0

                return 0.0

            global_env['f'] = f_constructor

        candidate_names = set()
        for proc in getattr(par, 'processes', []):
            candidates = []
            if isinstance(proc, hpc.PrefixProcess):
                candidates.append(proc)
            elif isinstance(proc, hpc.Sum):
                candidates.extend(get_sum_branches(proc))
            elif isinstance(proc, hpc.Replication):
                inner = getattr(proc, 'process', None) or getattr(proc, 'body', None)
                if isinstance(inner, hpc.PrefixProcess):
                    candidates.append(inner)
                elif isinstance(inner, hpc.Sum):
                    candidates.extend(get_sum_branches(inner))

            for cand in candidates:
                try:
                    if not (isinstance(cand, hpc.PrefixProcess) and isinstance(cand.prefix, hpc.Output)):
                        continue
                    out_pref = cand.prefix
                    ch_raw = channel_to_str(out_pref.channel)
                    m = re.match(r'^allocation\((.+)\)$', ch_raw)
                    if m:
                        candidate_names.add(m.group(1))
                    else:
                        candidate_names.add(normalize_channel_name(ch_raw))
                except Exception:
                    continue

        for name in sorted(candidate_names):
            if not name:
                continue
            try:
                if name in global_env:
                    continue
                const_val = extract_constant_from_sum(par, name)
                if const_val is not None:
                    try:
                        global_env[name] = float(const_val)
                    except Exception:
                        global_env[name] = const_val
            except Exception:
                continue
    except Exception as e:
        log_event(f"initialize_placeholders failed: {e}")

def register_function(name: str, fn: Any):
    """Register a function implementation (e.g., register_function).
    This allows calling the registered implementation and returning a numeric value when encountering f(...) in the simulation.
    """
    if name == 'f' and callable(fn):
        def _wrapped_f(*args, **kwargs):
            try:
                res = fn(*args, **kwargs)
                try:
                    log_event(f"f called with args={args} -> {res}")
                except Exception:
                    pass
                return res
            except Exception as e:
                try:
                    log_event(f"f raised exception: {e}")
                except Exception:
                    pass
                raise

        function_impls[name] = _wrapped_f
    else:
        function_impls[name] = fn


def extract_constant_from_sum(par: hpc.Parallel, var_name: str) -> Optional[float]:
    """Extract constant value from Sum process"""
    procs = getattr(par, 'processes', [par]) if not isinstance(par, hpc.Sum) else [par]

    def try_extract_from_prefix(pref_proc):
        pref = getattr(pref_proc, 'prefix', None)
        if isinstance(pref, hpc.Output):
            ch = channel_to_str(pref.channel)
            base_ch = normalize_channel_name(ch)
            if (var_name and (base_ch == var_name or ch == var_name)) or (not var_name):
                exprs = getattr(pref, 'expr_list', []) or getattr(pref, 'exprs', [])
                if exprs:
                    try:
                        return float(exprs[0])
                    except Exception:
                        val, missing = safe_eval_expr(str(exprs[0]), global_env)
                        if missing == set() and val is not None:
                            try:
                                return float(val)
                            except Exception:
                                return None
        return None

    for proc in procs:
        if isinstance(proc, hpc.Sum):
            branches = get_sum_branches(proc)
            for branch in branches:
                if isinstance(branch, hpc.PrefixProcess):
                    v = try_extract_from_prefix(branch)
                    if v is not None:
                        return v
        elif isinstance(proc, hpc.PrefixProcess):
            v = try_extract_from_prefix(proc)
            if v is not None:
                return v
        elif isinstance(proc, hpc.Replication):
            inner = getattr(proc, 'process', None) or getattr(proc, 'body', None)
            if inner is None:
                continue
            if isinstance(inner, hpc.Sum):
                for b in get_sum_branches(inner):
                    v = try_extract_from_prefix(b)
                    if v is not None:
                        return v
            elif isinstance(inner, hpc.PrefixProcess):
                v = try_extract_from_prefix(inner)
                if v is not None:
                    return v

    return None

# ---------- Reduction rules implementation ----------
def apply_reduction_rules(par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """
    Apply Reduction rules in order of priority:
    (a) [Pass] → (b) [Stop] → (c) [Comm] → (d) [Sense]/[Actuate] → (e) [Rep]
    """
    is_valid, message = validate_canonical_form(par)
    if not is_valid:
        log_event(f"Warning: Input process is not in canonical form: {message}")
    
    # (a) [Pass] rule: [B].P + M → P (condition true)
    applied, new_proc, log = apply_pass_rule(par)
    log_event(f"Attempted to apply [Pass] rule: applied={applied}, new_proc={new_proc}, log={log}")
    if applied:
        return True, new_proc, f"Applied [Pass] rule ({log})"
    
    # (b) [Stop] rule: terminate when continuous process boundary condition violated
    applied, new_proc, log = apply_stop_rule(par)
    log_event(f"Attempted to apply [Stop] rule: applied={applied}, new_proc={new_proc}, log={log}")
    if applied:
        return True, new_proc, f"Applied [Stop] rule ({log})"
    
    # (c) [Comm] rule: input-output communication
    applied, new_proc, log = apply_comm_rule(par)
    log_event(f"Attempted to apply [Comm] rule: applied={applied}, new_proc={new_proc}, log={log}")
    if applied:
        return True, new_proc, f"Applied [Comm] rule ({log})"
    
    # (d) [Sense] and [Actuate] rules: discrete-continuous interaction
    applied, new_proc, log = apply_sense_rule(par)
    log_event(f"Attempted to apply [Sense] rule: applied={applied}, new_proc={new_proc}, log={log}")
    if applied:
        return True, new_proc, f"Applied [Sense] rule ({log})"
    
    applied, new_proc, log = apply_actuate_rule(par)
    log_event(f"Attempted to apply [Actuate] rule: applied={applied}, new_proc={new_proc}, log={log}")
    if applied:
        return True, new_proc, f"Applied [Actuate] rule ({log})"
    
    # (e) [Rep] rule: replication process communication
    applied, new_proc, log = apply_rep_rule(par)
    log_event(f"Attempted to apply [Rep] rule: applied={applied}, new_proc={new_proc}, log={log}")
    if applied:
        return True, new_proc, f"Applied [Rep] rule ({log})"
    
    return False, par, None

def apply_pass_rule(par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """Apply [Pass] rule: [B].P + M → P"""
    new_processes = list(par.processes)
    applied = False
    log_msg = None
    applied_index = None
    
    for i, proc in enumerate(new_processes):
        if isinstance(proc, hpc.Sum):
            # Check if it is a regular Sum (not generated by Replication)
            if not _is_sum_from_replication(proc, par):
                applied, new_sum, msg = _apply_pass_to_sum(proc, par)
                if applied:
                    new_processes[i] = new_sum
                    log_msg = msg
                    applied_index = i
                    break
    
    if applied:
        new_par = to_strict_canonical_form_preserve_sums(hpc.Parallel(new_processes))
        try:
            log_event(f"PASS rule applied: idx={applied_index} msg={log_msg}")
        except Exception:
            pass
        return True, new_par, log_msg
    
    return False, par, None

def _is_sum_from_replication(sum_proc: hpc.Sum, par: hpc.Parallel) -> bool:
    """Determine if Sum process is from Replication"""
    def check_if_from_replication(proc, depth=0, max_depth=10):
        if depth > max_depth:
            return False
        if isinstance(proc, hpc.Replication):
            return True
        if isinstance(proc, hpc.PrefixProcess):
            return check_if_from_replication(proc.continuation, depth + 1, max_depth)
        elif isinstance(proc, hpc.Parallel):
            for sub_proc in proc.processes:
                if check_if_from_replication(sub_proc, depth + 1, max_depth):
                    return True
        elif isinstance(proc, hpc.Sum):
            for branch in get_sum_branches(proc):
                if check_if_from_replication(branch, depth + 1, max_depth):
                    return True
        elif isinstance(proc, hpc.Restriction):
            return check_if_from_replication(proc.process, depth + 1, max_depth)
        return False
    
    return check_if_from_replication(sum_proc)

def _apply_pass_to_sum(sum_proc: hpc.Sum, par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """Apply [Pass] rule to a regular Sum process"""
    branches = get_sum_branches(sum_proc)
    valid_branches = []
    applied_branch = None
    log_msg = None
    
    for branch in branches:
        if isinstance(branch, hpc.PrefixProcess) and isinstance(branch.prefix, hpc.Guard):
            condition = branch.prefix.condition
            cond_val, missing = safe_eval_expr(condition, global_env, par)
            
            if missing:
                valid_branches.append(branch)
            elif cond_val is True:
                if applied_branch is None:
                    applied_branch = branch
                    log_msg = f"Condition satisfied: {condition}"
                else:
                    valid_branches.append(branch)
            elif cond_val is False:
                continue
            else:
                valid_branches.append(branch)
        else:
            valid_branches.append(branch)
    
    if applied_branch:
        new_branches = [applied_branch.continuation] + valid_branches
        new_sum = _reconstruct_sum(new_branches)
        return True, new_sum, log_msg
    elif len(valid_branches) < len(branches):
        new_sum = _reconstruct_sum(valid_branches)
        return True, new_sum, f"Removed {len(branches)-len(valid_branches)} branches that did not satisfy the condition"
    
    return False, sum_proc, None

def _reconstruct_sum(branches: List[hpc.PrefixProcess]) -> Union[hpc.Sum, hpc.PrefixProcess]:
    """Reconstruct Sum process"""
    if len(branches) == 1:
        return branches[0]
    return hpc.Sum(branches=branches)

def apply_stop_rule(par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """Apply [Stop] rule: {e₀|v̇=e&B,RS}(y).P → P{e₀/y} (when boundary condition is false)"""
    new_processes = list(par.processes)
    
    for i, proc in enumerate(new_processes):
        if isinstance(proc, hpc.PrefixProcess) and isinstance(proc.prefix, hpc.Continuous):
            cont_prefix = proc.prefix
            ode = cont_prefix.ode
            
            local_env = {}
            for var, e0 in zip(ode.v, ode.e0):
                e0_val, _ = safe_eval_expr(str(e0), global_env)
                local_env[var] = e0_val

            try:
                local_env['t'] = global_env.get('t', 0.0)
            except Exception:
                local_env['t'] = 0.0

            for var, deriv in zip(ode.v, ode.e):
                primed = var + "'"
                try:
                    deriv_val, missing_deriv = safe_eval_expr(str(deriv), global_env, par)
                    if not missing_deriv:
                        local_env[primed] = deriv_val
                except Exception:
                    pass
            
            bound_val, missing = safe_eval_expr(str(ode.bound), local_env, par)
            if missing:
                continue
                
            if not bound_val:
                subst = dict(zip(ode.final_v or ode.v, ode.e0))
                for var, e0 in subst.items():
                    subst[var], _ = safe_eval_expr(str(e0), global_env)
                
                new_continuation = substitute_process(proc.continuation, subst)
                new_processes[i] = new_continuation
                
                new_par = to_strict_canonical_form_preserve_sums(hpc.Parallel(new_processes))
                log = f"Boundary condition violated: {ode.bound}, substitution={subst}"
                try:
                    log_event(f"STOP rule applied: idx={i} {log}")
                except Exception:
                    pass
                return True, new_par, log
    
    return False, par, None

def substitute_process(proc: hpc.Process, subst: Dict[str, Any]) -> hpc.Process:
    """Substitute variables in the process"""
    if proc is None:
        return hpc.Inaction()
    
    if isinstance(proc, hpc.Inaction):
        return proc
    
    if isinstance(proc, hpc.PrefixProcess):
        new_prefix = substitute_prefix(proc.prefix, subst)
        new_continuation = substitute_process(proc.continuation, subst)
        return hpc.PrefixProcess(new_prefix, new_continuation)
    
    if isinstance(proc, hpc.Parallel):
        new_procs = [substitute_process(p, subst) for p in proc.processes]
        return hpc.Parallel(new_procs)
    
    if isinstance(proc, hpc.Sum):
        new_branches = [substitute_process(b, subst) for b in get_sum_branches(proc)]
        return _reconstruct_sum(new_branches)
    
    if isinstance(proc, hpc.Replication):
        new_process = substitute_process(proc.process, subst)
        return hpc.Replication(new_process)
    
    if isinstance(proc, hpc.Restriction):
        new_process = substitute_process(proc.process, subst)
        return hpc.Restriction(proc.names, new_process)
    
    return proc

def substitute_prefix(prefix: hpc.Prefix, subst: Dict[str, Any]) -> hpc.Prefix:
    """Substitute expressions in the prefix"""
    if isinstance(prefix, hpc.Guard):
        new_cond = substitute_expr(prefix.condition, subst)
        return hpc.Guard(new_cond)
    elif isinstance(prefix, hpc.Output):
        new_exprs = [substitute_expr(e, subst) for e in prefix.expr_list]
        return hpc.Output(prefix.channel, new_exprs)
    return prefix

def substitute_expr(expr: str, subst: Dict[str, Any]) -> str:
    """Substitute variables in the expression"""
    if expr is None:
        return ""
    s = str(expr)
    for var, val in subst.items():
        s = re.sub(rf'\b{re.escape(var)}\b', str(val), s)
    return s

def apply_comm_rule(par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """Apply [Comm] rule: x(y).P + M || x̄⟨e⟩.Q + N → P{e/y} || Q"""
    inputs = []
    outputs = []
    
    for i, proc in enumerate(par.processes):
        if isinstance(proc, hpc.PrefixProcess):
            if isinstance(proc.prefix, hpc.Input):
                inputs.append((i, proc, proc, False, -1))
            elif isinstance(proc.prefix, hpc.Output):
                outputs.append((i, proc, proc, False, -1))
        elif isinstance(proc, hpc.Sum):
            branches = get_sum_branches(proc)
            for j, branch in enumerate(branches):
                if isinstance(branch, hpc.PrefixProcess):
                    if isinstance(branch.prefix, hpc.Input):
                        inputs.append((i, proc, branch, True, j))
                    elif isinstance(branch.prefix, hpc.Output):
                        outputs.append((i, proc, branch, True, j))
    for out_info in outputs:
        out_idx, out_parent, out_branch, out_is_sum, out_sum_idx = out_info
        out_prefix = out_branch.prefix
        out_ch = channel_to_str(out_prefix.channel)
        out_ch_norm = normalize_channel_name(out_ch)
        
        expr_values = []
        valid = True

        for expr in out_prefix.expr_list:
            val, missing = try_eval_output_expr(expr, par)
            if missing or val is None:
                break
            expr_values.append(val)
            if len(expr_values) == 1:
                global_env[out_ch_norm] = expr_values[0]
        if not valid:
            continue
        
        for in_info in inputs:
            in_idx, in_parent, in_branch, in_is_sum, in_sum_idx = in_info
            in_prefix = in_branch.prefix
            in_ch = channel_to_str(in_prefix.channel)
            in_ch_norm = normalize_channel_name(in_ch)
            if out_ch_norm == in_ch_norm and len(out_prefix.expr_list) == len(in_prefix.var_list):
                new_processes = list(par.processes)
                subst = dict(zip(in_prefix.var_list, expr_values))
                if not subst:
                    subst = dict(zip(in_prefix.var_list, out_prefix.expr_list))
                    global alias_map
                    alias_map = subst

                if in_is_sum:
                    in_branches = get_sum_branches(in_parent)
                    in_branches[in_sum_idx] = substitute_process(in_branch.continuation, subst)
                    new_processes[in_idx] = _reconstruct_sum(in_branches)
                else:
                    new_processes[in_idx] = substitute_process(in_branch.continuation, subst)

                if out_is_sum:
                    out_branches = get_sum_branches(out_parent)
                    out_branches[out_sum_idx] = out_branch.continuation
                    new_processes[out_idx] = _reconstruct_sum(out_branches)
                else:
                    new_processes[out_idx] = out_branch.continuation
                new_par = to_strict_canonical_form_preserve_sums(hpc.Parallel(new_processes))
                log = f"Communication channel({out_ch})(Communication message: {','.join(map(str, expr_values))})"
                try:
                    log_event(f"COMM consumption: out_idx={out_idx} in_idx={in_idx} channel={out_ch_norm} values={expr_values}")
                except Exception:
                    pass
                return True, new_par, log
    return False, par, None

def apply_sense_rule(par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """Apply [Sense] rule: v(y).P + M || {e0|...}(y).Q → P{e/y} || Q"""
    cont_items = collect_continuous_processes(par)
    discrete_inputs = []
    for i, proc in enumerate(par.processes):
        if isinstance(proc, hpc.PrefixProcess) and isinstance(proc.prefix, hpc.Input):
            discrete_inputs.append((i, proc, proc, False, -1))
        elif isinstance(proc, hpc.Sum):
            branches = get_sum_branches(proc)
            for j, branch in enumerate(branches):
                if isinstance(branch, hpc.PrefixProcess) and isinstance(branch.prefix, hpc.Input):
                    discrete_inputs.append((i, proc, branch, True, j))
 
    for cont_idx, cont_comp, ode, repl in cont_items:
        ready_set = set()
        try:
            ready_set = set(normalize_channel_name(channel_to_str(ch)) for ch in (ode.ready_set or []))
        except Exception:
            ready_set = set()
        if not ready_set:
            try:
                ready_set = set(normalize_channel_name(str(v)) for v in (getattr(ode, "v", []) or []))
            except Exception:
                ready_set = set()
        current_e0 = []
        for var_name in ode.v:
            val = None
            if (var_name in global_env) or (var_name in alias_map):
                val = global_env[var_name]
            else:
                norm = normalize_channel_name(str(var_name))
                if norm in global_env:
                   val = global_env[norm]
            if val is None:
                idx = ode.v.index(var_name)
                if idx < len(ode.e0):
                    v_, _ = safe_eval_expr(str(ode.e0[idx]), global_env)
                    val = v_ if v_ is not None else 0.0
                else:
                    val = 0.0
            current_e0.append(val)
 
        for in_info in discrete_inputs:
            in_idx, in_parent, in_branch, in_is_sum, in_sum_idx = in_info
            in_prefix = in_branch.prefix
            in_ch = normalize_channel_name(channel_to_str(in_prefix.channel))
            if ((in_ch in ready_set) or (in_ch in alias_map)) and len(in_prefix.var_list) == 1:
                ode_v_bases = [normalize_channel_name(str(v)) for v in ode.v]
                if (in_ch not in ode_v_bases) and (in_ch not in alias_map):
                    continue
                in_var = alias_map.get(in_ch, in_ch)
                var_idx = ode_v_bases.index(in_ch) if in_ch in ode_v_bases else ode_v_bases.index(in_var)
                if var_idx >= len(current_e0):
                    continue
                e_val = current_e0[var_idx]
                if e_val is None:
                    continue

                new_processes = list(par.processes)
                subst = {in_prefix.var_list[0]: e_val}
                if in_is_sum:
                    in_branches = get_sum_branches(in_parent)
                    in_branches[in_sum_idx] = substitute_process(in_branch.continuation, subst)
                    new_processes[in_idx] = _reconstruct_sum(in_branches)
                else:
                    new_processes[in_idx] = substitute_process(in_branch.continuation, subst)

                new_par = to_strict_canonical_form_preserve_sums(hpc.Parallel(new_processes))
                log = f"Sense channel matched ({in_ch}) value={e_val}"
                
                global_env[in_prefix.var_list[0]] = e_val
                return True, new_par, log
    
    return False, par, None

def apply_actuate_rule(par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """Apply [Actuate] rule: v̄⟨e⟩.P + M || {e0|...}(y).Q → P || {e'0|...}(y).Q"""
    cont_procs = []
    discrete_outputs = []
    
    for i, proc in enumerate(par.processes):
        if isinstance(proc, hpc.PrefixProcess) and isinstance(proc.prefix, hpc.Continuous):
            cont_procs.append((i, proc, proc.prefix.ode))
        elif isinstance(proc, hpc.PrefixProcess) and isinstance(proc.prefix, hpc.Output):
            discrete_outputs.append((i, proc, proc, False, -1))
        elif isinstance(proc, hpc.Sum):
            branches = get_sum_branches(proc)
            for j, branch in enumerate(branches):
                if isinstance(branch, hpc.PrefixProcess) and isinstance(branch.prefix, hpc.Output):
                    discrete_outputs.append((i, proc, branch, True, j))
    
    for cont_idx, cont_proc, ode in cont_procs:
        ready_set = set(normalize_channel_name(channel_to_str(ch)) for ch in ode.ready_set)
        current_e0 = [str(e) for e in ode.e0]
        
        for out_info in discrete_outputs:
            out_idx, out_parent, out_branch, out_is_sum, out_sum_idx = out_info
            out_prefix = out_branch.prefix
            out_ch = normalize_channel_name(channel_to_str(out_prefix.channel))

            if ((out_ch in ready_set) or (out_ch in alias_map)) and len(out_prefix.expr_list) == 1:
                e_val, missing = safe_eval_expr(str(out_prefix.expr_list[0]), global_env, par)
                if missing or e_val is None:
                    continue
                
                ode_v_bases = [normalize_channel_name(v) for v in ode.v]
                if (out_ch not in ode_v_bases) and (out_ch not in alias_map):
                    continue
                out_var = alias_map.get(out_ch, out_ch)
                log_event(f"Actuate output variable: {out_var}")
                var_idx = ode_v_bases.index(out_var) if out_ch in ode_v_bases else ode_v_bases.index(out_var)
                new_e0 = current_e0.copy()
                new_e0[var_idx] = str(e_val)
                
                new_ode = type(ode)(e0=new_e0, v=ode.v, e=ode.e, bound=ode.bound,
                                  ready_set=ode.ready_set, final_v=ode.final_v)
                new_cont_proc = hpc.PrefixProcess(
                    prefix=hpc.Continuous(new_ode),
                    continuation=cont_proc.continuation
                )
                
                new_processes = list(par.processes)
                new_processes[cont_idx] = new_cont_proc
                if out_is_sum:
                    out_branches = get_sum_branches(out_parent)
                    out_branches[out_sum_idx] = out_branch.continuation
                    new_processes[out_idx] = _reconstruct_sum(out_branches)
                else:
                    new_processes[out_idx] = out_branch.continuation
                
                new_par = to_strict_canonical_form_preserve_sums(hpc.Parallel(new_processes))
                log = f"Communication channel ({out_ch}) (communication message: {e_val}) - [Actuate] rule"
                return True, new_par, log
    
    return False, par, None

def apply_rep_rule(par: hpc.Parallel) -> Tuple[bool, hpc.Process, Optional[str]]:
    """Apply [Rep] rule: x̄⟨e⟩.P + M || !x(y).Q → P || Q{e/y} || !x(y).Q"""
    reps = []
    outputs = []
    
    for i, proc in enumerate(par.processes):
        if isinstance(proc, hpc.Replication):
            if isinstance(proc.process, hpc.PrefixProcess) and isinstance(proc.process.prefix, hpc.Input):
                reps.append((i, proc, proc.process.prefix))
        elif isinstance(proc, hpc.PrefixProcess) and isinstance(proc.prefix, hpc.Output):
            outputs.append((i, proc, proc, False, -1))
        elif isinstance(proc, hpc.Sum):
            branches = get_sum_branches(proc)
            for j, branch in enumerate(branches):
                if isinstance(branch, hpc.PrefixProcess) and isinstance(branch.prefix, hpc.Output):
                    outputs.append((i, proc, branch, True, j))
    
    for out_info in outputs:
        out_idx, out_parent, out_branch, out_is_sum, out_sum_idx = out_info
        out_prefix = out_branch.prefix
        out_ch = channel_to_str(out_prefix.channel)
        
        expr_values = []
        valid = True
        for expr in out_prefix.expr_list:
            val, missing = try_eval_output_expr(expr, par)
            if missing:
                inferred = resolve_variables_from_process(par, missing)
                if inferred:
                    temp_env = dict(global_env)
                    temp_env.update(inferred)
                    val2, miss2 = safe_eval_expr(str(expr), temp_env, par)
                    if not miss2 and val2 is not None:
                        val = val2
                        missing = set()
            if missing or val is None:
                valid = False
                break
            expr_values.append(val)

        if not valid:
            continue
        
        out_ch_norm = normalize_channel_name(out_ch)
        
        for rep_idx, rep_proc, rep_prefix in reps:
            rep_ch = channel_to_str(rep_prefix.channel)
            rep_ch_norm = normalize_channel_name(rep_ch)
            
            if out_ch_norm == rep_ch_norm and len(expr_values) == len(rep_prefix.var_list):
                new_processes = list(par.processes)
                subst = dict(zip(rep_prefix.var_list, expr_values))
                
                for key in list(subst.keys()):
                    if key in global_env:
                        val = global_env[key]
                    subst[key] = val
                new_instance = substitute_process(rep_proc.process.continuation, subst)
                
                if out_is_sum:
                    out_branches = get_sum_branches(out_parent)
                    out_branches[out_sum_idx] = out_branch.continuation
                    new_processes[out_idx] = _reconstruct_sum(out_branches)
                else:
                    new_processes[out_idx] = out_branch.continuation
                
                new_processes.append(new_instance)
                log_event(f"new_processes after Rep rule: {new_processes}")
                new_par = to_strict_canonical_form_preserve_sums(hpc.Parallel(new_processes))
                log = f"Communication channel ({out_ch}) (message: {','.join(map(str, expr_values))}) - [Rep] rule"
                return True, new_par, log
    return False, par, None

# ---------- Continuous rule implementation ----------
def apply_continuous_rules(par: hpc.Parallel, global_time: float) -> Tuple[hpc.Process, float, str]:
    """Synchronously evolve all continuous processes until the earliest boundary violation (Run strategy)"""
    cont_items = collect_continuous_processes(par)
    log_event(f"Collected {len(cont_items)} continuous processes for evolution")
    
    if not cont_items:
        return par, 0.0, "No continuous processes, skipping continuous evolution"

    per_results = []
    max_step = MAX_CONTINUOUS_STEP if MAX_CONTINUOUS_STEP is not None else 10.0
    
    for idx, comp, ode, repl in cont_items:
        if ode is None:
            continue
            
        vars_list = getattr(ode, "v", [])        
        derivs_list = getattr(ode, "e", [])
        bound_expr = getattr(ode, "bound", None)
        init_vals = getattr(ode, "e0", [])

        if len(vars_list) != len(derivs_list):
            import pdb; pdb.set_trace()
            log_event(f"Critical error: Variable list length ({len(vars_list)}) does not match derivative list length ({len(derivs_list)})!")
            log_event(f"Variable list: {vars_list}")
            log_event(f"Derivative list: {derivs_list}")
            log_event(f"ODE object: {ode}")
            log_event(f"ODE type: {type(ode)}")            
            log_event(f"ODE attribute list: {dir(ode)}")
            for attr in dir(ode):
                if not attr.startswith('_'):
                    try:
                        val = getattr(ode, attr)
                        log_event(f"  {attr}: {val} (类型: {type(val)})")
                    except:
                        pass
            
            raise ValueError(f"ODE system variable list and derivative list length mismatch: vars={len(vars_list)}, derivs={len(derivs_list)}")
        
        ode_info = {
            'idx': idx,
            'comp': comp,
            'ode': ode,
            'repl': repl,
            'vars_list': vars_list,
            'derivs_list': derivs_list,
            'bound_expr': bound_expr,
            'init_vals': init_vals
        }
        log_event(f"ODE system information idx={idx}: {ode_info}")

        y0 = []
        for i, var in enumerate(vars_list):
            log_event(f"Iterating initial values i: {i}, var: {var}, init_vals[i]: {init_vals[i]}")
            expr = init_vals[i] if i < len(init_vals) else "0.0"
            log_event(f"ODE initial value expression idx={idx}, var={var}: expr={expr}")
            v, missing = safe_eval_expr(str(expr), global_env)
            log_event(f"ODE initial value evaluation idx={idx}, var={var}: v={v}, missing={missing}")
            val = float(v) if (missing == set() and v is not None) else 0.0
            y0.append(val)
        log_event(f"ODE initial state idx={idx}: y0={y0}")

        def make_ode_fn(vars_list, derivs_list):
            def ode_fn(t, y):
                local = dict(global_env)
                local['t'] = global_time + t
                for j, vname in enumerate(vars_list):
                    if j < len(y):
                        local[vname] = y[j]
                dydt = []
                for expr in derivs_list:
                    try:
                        dv, missing = safe_eval_expr(str(expr), local)
                        if missing:
                            dv = 0.0
                        dv = 1.0 if isinstance(dv, bool) and dv else float(dv) if not isinstance(dv, bool) else 0.0
                    except Exception:
                        dv = 0.0
                    dydt.append(float(dv))
                return dydt
            return ode_fn

        ode_fn = make_ode_fn(vars_list, derivs_list)

        events = []
        if bound_expr:
            def make_event_fn(bound_expr):
                def event(t, y):
                    local = dict(global_env)
                    local['t'] = global_time + t
                    for j, vname in enumerate(vars_list):
                        if j < len(y):
                            local[vname] = y[j]
                    try:
                        bv, missing = safe_eval_expr(str(bound_expr), local)
                        if missing:
                            return 1.0
                        if isinstance(bv, bool):
                            return 1.0 if bv else -1.0
                        try:
                            return float(bv)
                        except Exception:
                            return 1.0
                    except Exception:
                        return 1.0
                event.terminal = True
                event.direction = -1
                return event
            events = [make_event_fn(bound_expr)]

        try:
            sol = solve_ivp(
                ode_fn,
                [0.0, max_step],
                y0,
                events=events if events else None,
                dense_output=True,
                rtol=1e-6,
                atol=1e-8,
                max_step=0.1
            )
        except Exception as e:
            sol = None

        event_time = None
        violated = False
        if sol is not None:
            if sol.t_events and len(sol.t_events) > 0 and sol.t_events[0].size > 0:
                event_time = float(sol.t_events[0][0])
                violated = True
            else:
                event_time = float(sol.t[-1]) if sol.t.size > 0 else 0.0
                violated = False
        else:
            event_time = 0.0
            violated = False

        per_results.append({
            'idx': idx,
            'comp': comp,
            'ode': ode,
            'repl': repl,
            'vars': vars_list,
            'y0': y0,
            'sol': sol,
            'event_time': event_time,
            'violated': violated
        })

    times = [r['event_time'] for r in per_results if r['event_time'] is not None]
    if not times:
        return par, 0.0, "No valid ODE solution"
    min_time = min(times)
    if min_time <= 1e-12:
        return par, 0.0, "No time advance in continuous evolution"

    new_processes = list(par.processes)
    final_env = {}
    for res in per_results:
        idx = res['idx']
        vars_list = res['vars']
        sol = res['sol']
        violated = res['violated']
        ode = res['ode']
        repl = res['repl']

        values_at_t = []
        if sol is not None and sol.t.size > 0:
            try:
                y_at = sol.sol(min_time)
                values_at_t = [float(y_at[j]) if j < y_at.shape[0] else 0.0 for j in range(len(vars_list))]
            except Exception:
                values_at_t = []
                for j in range(len(vars_list)):
                    if sol.y.shape[1] >= 1:
                        values_at_t.append(float(np.interp(min_time, sol.t, sol.y[j])))
                    else:
                        values_at_t.append(res['y0'][j] if j < len(res['y0']) else 0.0)
        else:
            values_at_t = [res['y0'][j] if j < len(res['y0']) else 0.0 for j in range(len(vars_list))]

        for vname, vval in zip(vars_list, values_at_t):
            final_env[vname] = vval
        if res['violated'] and res['event_time'] <= min_time + 1e-9:
            substitution = {v: str(final_env.get(v, 0.0)) for v in vars_list}
            new_comp = substitute_process(repl, substitution)
            if idx < len(new_processes):
                new_processes[idx] = replace_first_continuous(new_processes[idx], new_comp)
        else:
            new_ode = copy.deepcopy(ode)
            new_ode.e0 = [str(final_env.get(v, 0.0)) for v in vars_list]
            new_comp = hpc.PrefixProcess(prefix=hpc.Continuous(new_ode), continuation=repl)
            if idx < len(new_processes):
                new_processes[idx] = replace_first_continuous(new_processes[idx], new_comp)

    new_par = to_strict_canonical_form_preserve_sums(hpc.Parallel(new_processes))

    try:
        visible = []
        for r in per_results:
            for v in r.get('vars', []):
                if v not in visible and v not in hidden_vars:
                    visible.append(v)
        log_trace("global_time," + ",".join(visible))
        if min_time > 0:
            n_samples = min(10, max(1, int(np.ceil(min_time / 0.01))))
            sample_dt = max(min_time / n_samples, min(0.01, min_time))
            t_samples = list(np.arange(0.0, min_time, sample_dt)) + [min_time]
        else:
            t_samples = [0.0]

        for t_rel in t_samples:
            snapshot = {}
            for r in per_results:
                vars_list = r.get('vars', [])
                sol = r.get('sol', None)
                y0 = r.get('y0', [])
                if sol is not None and getattr(sol, "t", None) is not None and sol.t.size > 0:
                    try:
                        y_at = sol.sol(t_rel)
                        for j, v in enumerate(vars_list):
                            try:
                                snapshot[v] = float(y_at[j]) if j < y_at.shape[0] else float(y0[j] if j < len(y0) else 0.0)
                            except Exception:
                                snapshot[v] = float(y0[j] if j < len(y0) else 0.0)
                    except Exception:
                        for j, v in enumerate(vars_list):
                            try:
                                snapshot[v] = float(np.interp(t_rel, sol.t, sol.y[j])) if sol is not None and sol.t.size > 0 else float(y0[j] if j < len(y0) else 0.0)
                            except Exception:
                                snapshot[v] = float(y0[j] if j < len(y0) else 0.0)
                else:
                    for j, v in enumerate(vars_list):
                        snapshot[v] = float(y0[j] if j < len(y0) else 0.0)

            trace_line = format_trace_line(global_time + t_rel, snapshot, visible)
            log_trace(trace_line)
    except Exception as e:
        log_event(f"Failed to record trace: {e}")

    return new_par, float(min_time), f"Continuous evolution: minimum advance {min_time:.6f} seconds"

def simulate_block(block: Block, max_steps: int = 100000, init_state: dict = None):
    """Main simulation function"""
    global global_env
    
    global_env = init_state or {}
    reset_log_files()
    
    log_event("=== Starting HPC continuous process simulation ===")
    
    par, defs = prepare_system_from_block(block)
    
    step = 0
    global_time = 0.0
    
    while step < max_steps:
        step += 1
        log_event(f"\n--- Step {step} ---")
        
        pretty_print_state(par, step, global_time)
        
        reduced, new_par, event_log = apply_reduction_rules(par)
        if reduced:
            par = new_par
            log_event(f"Discrete reduction: {event_log}")
            
            if is_terminated(par):
                log_event("System terminated")
                break
            continue
        
        if any_continuous_process(par):
            try:
                new_par, evolve_time, trace_log = apply_continuous_rules(par, global_time)
                
                if evolve_time > 0:
                    par = new_par
                    global_time += evolve_time
                    log_event(f"Continuous evolution completed: advanced {evolve_time:.6f} seconds")
                    
                    log_event(f"Checking {par}")
                    if is_terminated(par):
                        log_event("System terminated (after continuous evolution)")
                        break
                    continue
                else:
                    log_event("Continuous evolution with no time advance")
            except Exception as e:
                log_event(f"Continuous evolution error: {e}")
        
        if is_terminated(par):
            log_event("System terminated")
            break
        
        if step >= max_steps:
            log_event("Reached maximum step limit")
            break
    pretty_print_state(par, step, global_time)
    log_event(f"=== Simulation ended: {step} steps, total time {global_time:.6f} seconds ===")
    flush_logs()

def collect_continuous_processes(par: hpc.Parallel) -> List[Tuple[int, Any, Any, Any]]:
    """Collect top-level continuous processes - C processes"""
    cont_items = []

    def find_continuous_in_node(node):
        """Recursively find the first Continuous prefix in the node, return (found_prefix_owner, ode, continuation) or (None, None, None)"""
        if node is None:
            return None, None, None

        if isinstance(node, hpc.PrefixProcess) and isinstance(getattr(node, "prefix", None), hpc.Continuous):
            return node, node.prefix.ode, node.continuation

        return None, None, None

    for idx, comp in enumerate(getattr(par, "processes", [])):
        try:
            found, ode, repl = find_continuous_in_node(comp)
            if found is not None and ode is not None:
                cont_items.append((idx, comp, ode, repl))
        except Exception:
            continue

    return cont_items

def format_trace_line(t: float, env: Dict[str, float], visible_vars: List[str]) -> str:
    """Format trace data line"""
    values = [f"{t:.6f}"]
    for var in visible_vars:
        val = env.get(var, 0.0)
        values.append(f"{float(val):.6f}")
    return ",".join(values)

def update_global_environment(env: Dict[str, float]):
    """Update global environment"""
    global global_env
    for key, value in env.items():
        if not key.startswith('_'):
            global_env[key] = value


def compile_expr_callable(expr: str, var_to_idx: Dict[str, int], const_env: Dict[str, float], par: Optional[hpc.Parallel] = None):
    """Compile expression into a fast callable object"""
    s = str(expr).strip()
    if s == "":
        return (lambda t, y: False, s)
    
    ids = identifiers_in_expr(s)
    expr2 = s
    
    for ident in sorted(ids, key=lambda x: -len(x)):
        if ident == 't':
            continue
        if ident in var_to_idx:
            expr2 = re.sub(rf'\b{re.escape(ident)}\b', f'y[{var_to_idx[ident]}]', expr2)
        elif ident in const_env:
            val = const_env[ident]
            if isinstance(val, (int, float, bool)):
                expr2 = re.sub(rf'\b{re.escape(ident)}\b', repr(val), expr2)
    
    expr2 = expr2.replace("^", "**").replace("¬", " not ").replace("&&", " and ").replace("||", " or ")
    
    try:
        code_obj = compile(expr2, f"<expr:{expr}>", "eval")
    except Exception:
        def slow_fn(t, y):
            y_dict = {k: y[i] for k, i in var_to_idx.items()}
            val, missing = safe_eval_expr(s, y_dict, par)
            return val
        return (slow_fn, s)
    
    def fn(t, y):
        try:
            local_vars = {"t": t, "y": y}
            for ident in ids:
                if ident == 't':
                    continue
                if ident in var_to_idx:
                    continue
                if ident in global_env:
                    try:
                        local_vars[ident] = global_env[ident]
                    except Exception:
                        pass
            return eval(code_obj, {"__builtins__": {"abs": abs, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos}}, local_vars)
        except Exception:
            y_dict = {k: y[i] for k, i in var_to_idx.items()}
            val, missing = safe_eval_expr(s, y_dict, par)
            return val
    
    return (fn, s)

def any_continuous_process(par: hpc.Parallel) -> bool:
    """Check if there is any continuous process (including reduced 0 processes)"""
    if not isinstance(par, hpc.Parallel):
        return False
    
    for proc in par.processes:
        if (isinstance(proc, hpc.PrefixProcess) and 
            isinstance(proc.prefix, hpc.Continuous)):
            return True
        
        if has_continuous_structure(proc):
            return True
    
    return False

def has_continuous_structure(proc: hpc.Process) -> bool:
    """Recursively check if the process contains continuous structures"""
    if proc is None:
        return False
    
    if isinstance(proc, hpc.PrefixProcess):
        if isinstance(proc.prefix, hpc.Continuous):
            return True
        return has_continuous_structure(proc.continuation)
    
    elif isinstance(proc, hpc.Parallel):
        for sub_proc in proc.processes:
            if has_continuous_structure(sub_proc):
                return True
    
    elif isinstance(proc, hpc.Sum):
        for branch in get_sum_branches(proc):
            if has_continuous_structure(branch):
                return True
    
    elif isinstance(proc, hpc.Replication):
        return has_continuous_structure(proc.process)
    
    elif isinstance(proc, hpc.Restriction):
        return has_continuous_structure(proc.process)
    
    return False

def is_terminated(proc: hpc.Process) -> bool:
    """Check if the process is terminated.
    Termination conditions:
      - proc is None or Inaction
      - or top-level is Parallel, and all subprocesses are Replication or Inaction
    """
    log_event(f"Check process termination: {proc}")
    if proc is None:
        return True

    if isinstance(proc, hpc.Inaction):
        return True

    if isinstance(proc, hpc.Parallel):
        for sub in getattr(proc, "processes", []):
            if isinstance(sub, hpc.Inaction):
                continue
            if isinstance(sub, hpc.Replication):
                continue
            t = classify_process_type(sub)
            if t not in ("R"):
                return False
        return True

    try:
        canonical = to_strict_canonical_form_preserve_sums(proc)
        return isinstance(canonical, hpc.Inaction)
    except Exception:
        return False

def update_no_progress_count(par: hpc.Parallel, global_time: float, last_par_str: str, 
                           last_time_snapshot: float, current_count: int) -> int:
    """Update no-progress counter"""
    try:
        cur_par_str = str(par)
        if cur_par_str == last_par_str and abs(global_time - last_time_snapshot) < 1e-12:
            return current_count + 1
        else:
            return 0
    except Exception:
        return current_count

def reset_log_files():
    """Reset log files"""
    global events_buffer, trace_buffer
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        f.write("# HpC simulation event log\n")
    with open(TRACE_FILE, "w", encoding="utf-8") as f:
        f.write("# Continuous evolution trace log\n")
    with open(STEP_FILE, "w", encoding="utf-8") as f:
        f.write("# Step log\n")
    events_buffer = []
    trace_buffer = []
    step_buffer = []

def log_event(s: str):
    """Log event to event log file"""
    global events_buffer
    events_buffer.append(s)
    if len(events_buffer) >= LOG_FLUSH_THRESHOLD:
        flush_logs()

def log_trace(s: str):
    """Log continuous evolution trace data"""
    global trace_buffer
    trace_buffer.append(s)
    if len(trace_buffer) >= LOG_FLUSH_THRESHOLD:
        flush_logs()

def log_step(s: str):
    """Log event to step log file"""
    global step_buffer
    step_buffer.append(s)
    if len(step_buffer) >= LOG_FLUSH_THRESHOLD:
        flush_logs()

def pretty_print_state(par: hpc.Parallel, step: int, global_time: float = None):
    """Print the current simulation state"""
    log_step(f"\n=== STEP {step} time: {global_time:.6f} ===")
    
    m_count = 0
    c_count = 0
    r_count = 0
    other_count = 0
    
    for i, proc in enumerate(par.processes):
        proc_type = classify_process_type(proc)
        if proc_type == "M":
            m_count += 1
            type_symbol = "M"
        elif proc_type == "C":
            c_count += 1
            type_symbol = "C"
        elif proc_type == "R":
            r_count += 1
            type_symbol = "R"
        else:
            other_count += 1
            type_symbol = "inaction"
        
        s = str(proc)
        log_step(f"  [{i}] {type_symbol}: {s}")
    
    log_step(f"Process counts: M={m_count}, C={c_count}, R={r_count}, Others={other_count}")
    log_step("================================\n")

# ---------- Compatibility functions ----------
def evaluate_bound_condition(bound_expr: str, vars_dict: Dict[str, float]) -> bool:
    """Compatibility with old version: evaluate boundary condition"""
    val, _ = safe_eval_expr(bound_expr, vars_dict)
    return bool(val)

def run_continuous_evolution(par: hpc.Parallel, dt: float = 0.1) -> Tuple[float, hpc.Parallel]:
    """Compatibility with old version: continuous evolution"""
    try:
        new_par, evolve_time, _ = apply_continuous_rules(par, 0.0)
        return evolve_time, new_par
    except:
        return 0.0, par


def get_process_structure(process, indent=0, seen=None):
    """Recursively get the structure information of the process, supporting all types defined in hpc"""
    if seen is None:
        seen = set()
    
    process_id = id(process)
    if process_id in seen:
        return f"{'  ' * indent}[Circular reference: {type(process).__name__}]"
    seen.add(process_id)
    
    indent_str = "  " * indent
    result = []
    
    if isinstance(process, hpc.Wait):
        result.append(f"{indent_str}└── Wait({process.delay})")
    
    elif isinstance(process, hpc.Tau):
        result.append(f"{indent_str}└── Tau")
    
    elif isinstance(process, hpc.Channel):
        para_str = f"({process.para})" if process.para else ""
        result.append(f"{indent_str}└── {type(process).__name__}({process.name}{para_str})")
    
    elif isinstance(process, hpc.Input):
        channel_str = get_process_structure(process.channel, indent + 1, seen).split('\n', 1)[0]
        var_list_str = ", ".join(process.var_list)
        result.append(f"{indent_str}└── Input(channel={channel_str}, vars=[{var_list_str}])")
    
    elif isinstance(process, hpc.Output):
        channel_str = get_process_structure(process.channel, indent + 1, seen).split('\n', 1)[0]
        expr_list_str = ", ".join(str(e) for e in process.expr_list)
        result.append(f"{indent_str}└── Output(channel={channel_str}, exprs=[{expr_list_str}])")
    
    elif isinstance(process, hpc.Guard):
        result.append(f"{indent_str}└── Guard(condition='{process.condition}')")
    
    elif isinstance(process, hpc.ODE):
        result.append(f"{indent_str}└── ODE")
        result.append(f"{indent_str}    ├── Initial values: {process.e0}")
        result.append(f"{indent_str}    ├── Variables: {process.v}")
        result.append(f"{indent_str}    ├── Derivatives: {process.e}")
        result.append(f"{indent_str}    ├── Boundary conditions: {process.bound}")
        result.append(f"{indent_str}    ├── Ready channels: {[str(ch) for ch in process.ready_set]}")
        result.append(f"{indent_str}    └── Final variables: {process.final_v}")
    
    elif isinstance(process, hpc.Continuous):
        result.append(f"{indent_str}└── Continuous")
        result.append(get_process_structure(process.ode, indent + 1, seen))
    
    elif isinstance(process, hpc.Assignment):
        result.append(f"{indent_str}└── Assignment({process.var} := {process.expr})")
    
    elif isinstance(process, hpc.Inaction):
        result.append(f"{indent_str}└── Inaction")
    
    elif isinstance(process, hpc.PrefixProcess):
        result.append(f"{indent_str}└── PrefixProcess")
        result.append(f"{indent_str}    ├── prefix:")
        result.append(get_process_structure(process.prefix, indent + 2, seen))
        result.append(f"{indent_str}    └── continuation:")
        result.append(get_process_structure(process.continuation, indent + 2, seen))
    
    elif isinstance(process, hpc.Parallel):
        result.append(f"{indent_str}└── Parallel (number of processes: {len(process.processes)})")
        for i, sub_process in enumerate(process.processes):
            result.append(f"{indent_str}    ├── Subprocess[{i}]:")
            result.append(get_process_structure(sub_process, indent + 2, seen))
    
    elif isinstance(process, hpc.Sum):
        branches = get_sum_branches(process)
        result.append(f"{indent_str}└── Sum (number of branches: {len(branches)})")
        for i, branch in enumerate(branches):
            result.append(f"{indent_str}    ├── Branch[{i}]:")
            result.append(get_process_structure(branch, indent + 2, seen))
    
    elif isinstance(process, hpc.Restriction):
        result.append(f"{indent_str}└── Restriction(ν {', '.join(process.names)})")
        result.append(get_process_structure(process.process, indent + 1, seen))
    
    elif isinstance(process, hpc.Replication):
        result.append(f"{indent_str}└── Replication")
        result.append(get_process_structure(process.process, indent + 1, seen))
    
    elif isinstance(process, hpc.NamedProcess):
        result.append(f"{indent_str}└── NamedProcess({process.name})")
        result.append(get_process_structure(process.body, indent + 1, seen))
    
    elif isinstance(process, hpc.ProcessVariable):
        result.append(f"{indent_str}└── ProcessVariable({process.name})")
    
    elif isinstance(process, hpc.Recursion):
        params_str = f"({','.join(process.params)})" if process.params else ""
        actual_str = f"@<{','.join(process.actual_params)}>" if process.actual_params else ""
        result.append(f"{indent_str}└── Recursion(μ{process.var}{params_str}{actual_str})")
        result.append(get_process_structure(process.proc, indent + 1, seen))
    
    elif isinstance(process, hpc.Conditional):
        result.append(f"{indent_str}└── Conditional(if {process.condition})")
        result.append(f"{indent_str}    ├── true branch:")
        result.append(get_process_structure(process.branch0, indent + 2, seen))
        result.append(f"{indent_str}    └── false branch:")
        result.append(get_process_structure(process.branch1, indent + 2, seen))
    
    elif isinstance(process, hpc.Loop):
        result.append(f"{indent_str}└── Loop(μ{process.channel})")
        result.append(f"{indent_str}    ├── formal parameters: {process.formal_paras}")
        result.append(f"{indent_str}    ├── actual parameters: {process.actual_paras}")
        result.append(get_process_structure(process.process, indent + 1, seen))
    
    elif isinstance(process, hpc.System):
        result.append(f"{indent_str}└── System(restricted names: {process.restricted_names})")
        result.append(get_process_structure(process.process, indent + 1, seen))
    
    elif isinstance(process, Block):
        result.append(f"{indent_str}└── Block (number of definitions: {len(process.definitions)})")
        for i, definition in enumerate(process.definitions):
            result.append(f"{indent_str}    ├── Definition[{i}]:")
            result.append(get_process_structure(definition, indent + 2, seen))
    
    else:
        result.append(f"{indent_str}└── {type(process).__name__} (unknown type)")
        for attr_name in ['process', 'body', 'continuation', 'branches', 'choices', 'summands', 'terms']:
            if hasattr(process, attr_name):
                attr_value = getattr(process, attr_name)
                if attr_value is not None and isinstance(attr_value, (hpc.Process, Block)):
                    result.append(f"{indent_str}    ├── {attr_name}:")
                    result.append(get_process_structure(attr_value, indent + 2, seen))
    
    return "\n".join(result)

def analyze_process_structure(process):
    """Analyze the process structure and return detailed information"""
    structure_info = {
        'type_counts': {},
        'max_depth': 0,
        'total_nodes': 0,
        'has_cycles': False
    }
    
    def count_types(proc, depth=0, seen=None):
        if seen is None:
            seen = set()
        
        process_id = id(proc)
        if process_id in seen:
            structure_info['has_cycles'] = True
            return depth
        
        seen.add(process_id)
        structure_info['max_depth'] = max(structure_info['max_depth'], depth)
        structure_info['total_nodes'] += 1
        
        proc_type = type(proc).__name__
        structure_info['type_counts'][proc_type] = structure_info['type_counts'].get(proc_type, 0) + 1
        
        max_child_depth = depth
        
        if isinstance(proc, (hpc.PrefixProcess, hpc.Replication, hpc.Restriction, hpc.NamedProcess)):
            if hasattr(proc, 'continuation'):
                max_child_depth = max(max_child_depth, count_types(proc.continuation, depth + 1, seen))
            if hasattr(proc, 'process'):
                max_child_depth = max(max_child_depth, count_types(proc.process, depth + 1, seen))
            if hasattr(proc, 'body'):
                max_child_depth = max(max_child_depth, count_types(proc.body, depth + 1, seen))
        
        elif isinstance(proc, hpc.Parallel):
            for sub_proc in proc.processes:
                max_child_depth = max(max_child_depth, count_types(sub_proc, depth + 1, seen))
        
        elif isinstance(proc, hpc.Sum):
            for branch in get_sum_branches(proc):
                max_child_depth = max(max_child_depth, count_types(branch, depth + 1, seen))
        
        elif isinstance(proc, Block):
            for definition in proc.definitions:
                max_child_depth = max(max_child_depth, count_types(definition, depth + 1, seen))
        
        return max_child_depth
    
    count_types(process)
    return structure_info

def print_process_structure(process, title="Process Structure"):
    """Print the complete structure information of the process"""
    print(f"\n=== {title} ===")
    print(get_process_structure(process))
    
    structure_info = analyze_process_structure(process)
    print(f"\n=== Structure Analysis ===")
    print(f"Total nodes: {structure_info['total_nodes']}")
    print(f"Maximum depth: {structure_info['max_depth']}")
    print(f"Contains cycles: {'Yes' if structure_info['has_cycles'] else 'No'}")
    print("\nType counts:")
    for proc_type, count in sorted(structure_info['type_counts'].items()):
        print(f"  {proc_type}: {count}")

if __name__ == "__main__":
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()

    root = parse_example(example)
    print("=== Parsing Result ===")
    print_fragment(root)
    
    role_variables = collect_role_variables(root)
    print("\n=== Role Variable Collection Results ===")
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

    print("\n=== t Translation Result ===")
    print(translated)

    with open("translated_output.txt", "w", encoding="utf-8") as f:
        f.write(str(translated))
        f.write("\n")
    
    standard_form = to_standard_form(translated)
    standard_str = str(standard_form)

    print("\n=== Standard Form Result ===")
    print(standard_str)

    with open("standardized_output.txt", "w", encoding="utf-8") as f:
        f.write(standard_str)

    print_process_structure(standard_form, "Standard Form Process Structure")

    def _parse_numeric(x):
        """Try to parse x as a numeric value; return (value, True) or (None, False)"""
        try:
            if isinstance(x, (int, float)):
                return float(x), True
            if isinstance(x, ChannelPayload):
                if len(x.args) > 0:
                    return _parse_numeric(x.args[0])
                return None, False
            val, missing = safe_eval_expr(str(x), global_env)
            if missing:
                return None, False
            if val is None:
                return None, False
            return float(val), True
        except Exception:
            return None, False

    def f_safe(p0, v0, endpoint, control_dt: float = 1.0, a_max: float = 1.0, a_min: float = -1.0, v_limit: float = 40.0):
        """
        Implement f using user-provided safe acceleration logic.
        """
        p0n, ok_p = _parse_numeric(p0)
        v0n, ok_v = _parse_numeric(v0)
        endn, ok_e = _parse_numeric(endpoint)
        log_event(f"f_safe call: p0={p0} (ok={ok_p}), v0={v0} (ok={ok_v}), endpoint={endpoint} (ok={ok_e})")

        if not (ok_p and ok_v and ok_e):
            return ChannelPayload('f', (p0, v0, endpoint))

        p = float(p0n)
        v = float(v0n)
        endpoint_f = float(endn)

        MAX_ACCELERATION = float(a_max)
        MAX_DECELERATION = float(a_min)
        MAX_VELOCITY = float(v_limit) 
        CONTROL_PERIOD = float(control_dt)
        SAFETY_MARGIN = 2.0

        if p < 9000:
            return float(MAX_ACCELERATION)
    try:
        register_function('f', f_safe)
        print("registered function 'f' -> f_safe(p0,v0,endpoint)")
    except Exception as e:
        print(f"Failed to register 'f': {e}")

    simulate_block(standard_form, max_steps=1200)
    print("=== Simulation Complete ===")