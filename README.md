# Analysis and Verification of Mobile and Cyber-Physical Extensions to Sequence Diagrams in the Hybrid $\pi$-Calculus
This project implements a tool for automatically translating Extended Sequence Diagrams (ESD) into Hybrid $\pi$-Calculus (HpC) processes, enabling modeling, simulation, and analysis. The tool supports formal modeling of complex discrete-continuous hybrid systems, process standardization, automated simulation, and trajectory visualization. 

## File Structure

| File | Description |
|------|-------------|
| `README.md` | Project documentation |
| `esd.py` | Core module for ESD parsing and HpC process generation |
| `hpc.py` | Syntax structure definition for Hybrid $\pi$-Calculus |
| `expr.py` | Expression evaluation module supporting variables, constants, binary operations (e.g., `+`, `-`, `*`, `/`), and unary operations (e.g., `¬` for logical negation) |
| `standardize_process.py` | Standardization of HpC processes and variable conflict resolution |
| `simulator.py` | Discrete-continuous hybrid simulation engine |
| `sim_test.py` | Batch automation testing script for multiple cases |
| `simulatorplot.py` | Automated trajectory plotting and statistical analysis |
| `example/` | Directory for batch case inputs and outputs |

## Usage

1. Prepare the input file `example.txt` following the specified sequence diagram format.
2. Run the main script `esd.py`:
   ```bash
   python esd.py
   ```
3. View the translation results:
   - Console output
   - The generated `example_translated_output.txt` file (HpC process with syntactic sugar)
   - The generated `example_standardized_output.txt` file (standardized HpC processes)
4. Run the simulation script `simulator.py`:
   ```bash
   python simulator.py
   ```
   This generates the following simulation outputs:
   - `example_simulation_events.txt` – Simulation event log
   - `example_steps.txt` – Reduction step log
   - `example_trace.txt` – Continuous evolution trajectory file
5. Visualize the trajectories. Run the visualization script `simulatorplot.py`:
   ```bash
   python simulatorplot.py
   ```
   This produces a variable evolution plot, as shown below:
![alt text](example_trace_except_p.png)
6. Run batch tests. Execute the batch testing script `sim_test.py`:
   ```bash
   python sim_test.py
   ```
   This automatically processes multiple case files in the `example/` directory.

## Example

Input (`example.txt`):
```
@startuml
Train -> Train: terminus := 10000
LeftSector -> LeftSector: handover_point := 4000
LeftSector -> LeftSector: endpoint := 5000
RightSector -> RightSector: handover_point := 9000
RightSector -> RightSector: endpoint := 10000
activate Train
note over of Train: <<ode>> {0, 0, 0 | p_dot=v, v_dot=a, a_dot=0 & p<terminus and v>=0}
LeftSector -> Train: <<sense>> p0 := p
loop [1] LeftSector: p0 < handover_point
  LeftSector -> Train: <<sense>> v0 := v
  LeftSector -> LeftSector: a0 := f(p0, v0, endpoint)
  LeftSector -> Train: <<actuate>> a := a0
  activate LeftSector
  note right of LeftSector: delay(1)
  deactivate LeftSector
  LeftSector -> Train: <<sense>> p0 := p
end
LeftSector -> RightSector: handover
RightSector -> LeftSector: yes
LeftSector -> RightSector: channels(p1, v1, a1 := p, v, a)
RightSector -> Train: <<sense>> p0 := p1
loop [2] RightSector: p0 < handover_point
  RightSector -> Train: <<sense>> v0 := v1
  RightSector -> RightSector: a0 := f(p0, v0, endpoint)
  RightSector -> Train: <<actuate>> a1 := a0
  activate RightSector
  note left of RightSector: delay(1)
  deactivate RightSector
  RightSector -> Train: <<sense>> p0 := p1
end
RightSector -> Train: <<actuate>> a1 := -1
deactivate Train
@enduml
```

Translated output (`example_translated_output.txt`):
```
Train ::= ⟨terminus:=10000⟩.{0, 0, 0 | p_dot=v, v_dot=a, a_dot=0 & p<terminus and v>=0, {p̅, v̅, a}}.0 || LeftSector ::= ⟨handover_point_1:=4000⟩.⟨endpoint_1:=5000⟩.p(p0).(ν loop[1]) loop[1]⟨p0, handover_point_1⟩.0 || !loop[1](p0, handover_point_1).([p0 < handover_point_1].v(v0).⟨a0_1:=f(p0, v0, endpoint_1)⟩.a̅⟨a0_1⟩.(ν t) {0 | t_dot=1 & t<1.0, {}}.p(p0).loop[1]⟨p0, handover_point_1⟩.0 + [(¬p0 < handover_point_1)].handover̅⟨⟩.yes().channels̅⟨p, v, a⟩.0) || RightSector ::= ⟨handover_point_2:=9000⟩.⟨endpoint_2:=10000⟩.handover().yes̅⟨⟩.channels(p1, v1, a1).p1(p0).(ν loop[2]) loop[2]⟨p0, handover_point_2⟩.0 || !loop[2](p0, handover_point_2).([p0 < handover_point_2].v1(v0).⟨a0_2:=f(p0, v0, endpoint_2)⟩.a1̅⟨a0_2⟩.(ν t) {0 | t_dot=1 & t<1.0, {}}.p1(p0).loop[2]⟨p0, handover_point_2⟩.0 + [(¬p0 < handover_point_2)].a1̅⟨-1⟩.0)
```

Standardized output (`example_standardized_output.txt`):
```
Train ::= ⟨terminus:=10000⟩.{0, 0, 0 | p_dot=v, v_dot=a, a_dot=0 & p<terminus and v>=0, {p̅, v̅, a}}.0
LeftSector ::= ⟨handover_point_1:=4000⟩.⟨endpoint_1:=5000⟩.p(p0).loop[1]⟨p0, handover_point_1⟩.0
Replication 1 ::= !loop[1](p0, handover_point_1).([p0 < handover_point_1].v(v0).⟨a0_1:=f(p0, v0, endpoint_1)⟩.a̅⟨a0_1⟩.{0 | t_dot=1 & t<1.0, {}}.p(p0).loop[1]⟨p0, handover_point_1⟩.0 + [(¬p0 < handover_point_1)].handover̅⟨⟩.yes().channels̅⟨p, v, a⟩.0)
RightSector ::= ⟨handover_point_2:=9000⟩.⟨endpoint_2:=10000⟩.handover().yes̅⟨⟩.channels(p1, v1, a1).p1(p0).loop[2]⟨p0, handover_point_2⟩.0
Replication 2 ::= !loop[2](p0, handover_point_2).([p0 < handover_point_2].v1(v0).⟨a0_2:=f(p0, v0, endpoint_2)⟩.a1̅⟨a0_2⟩.{0 | t'_dot=1 & t'<1.0, {}}.p1(p0).loop[2]⟨p0, handover_point_2⟩.0 + [(¬p0 < handover_point_2)].a1̅⟨-1⟩.0)
System ::= (ν loop[1]) (ν loop[2]) (ν t) (ν t') Train || LeftSector || Replication 1 || RightSector || Replication 2
```
## Development Environment

Python 3.9.18
typing_extensions 4.12.2
scipy 1.13.1
pandas 2.2.3
matplotlib 3.9.4.
