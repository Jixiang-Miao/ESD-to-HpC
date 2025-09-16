# Modeling and Analysis of Cyber-Physical Systems in the Hybrid π-Calculus Using Extended Sequence Diagrams
This project implements a tool that converts Extended Sequence Diagram (ESD) into Hybrid π-Calculus (HpC) processes. The tool parses ESD written in a specific format and translates them into formal HpC process expressions.

## File Structure

- `hpc.py`: Defines the syntax of HpC, including prefixes and process types.
- `esd.py`: Contains the logic for converting ESD to HpC, including parsing and transformation of different fragment types.
- `example.txt`: Input file containing the sequence diagram description.
- `translated_output.txt`: Output file containing the translated HpC process with syntactic sugar.
- `standardize_process.py.txt`: Translates HpC processes with syntactic sugar to standardized HpC processes.
- `standardized_output.txt`: Output file containing the standardized HpC process output.

## Usage

1. Prepare the input file `example.txt` following the specified sequence diagram format.
2. Run the main script `esd.py`:
   ```bash
   python esd.py
   ```
3. View the translation results:
   - Console output
   - The generated `translated_output.txt` file (HpC process with syntactic sugar)
   - The generated `standardized_output.txt` file (standardized HpC process)

## Example

Input(`example.txt`):
```
@startuml
Train -> Train: terminus := 10000
LeftSector -> LeftSector: handover_point := 4000
LeftSector -> LeftSector: endpoint := 5000
RightSector -> RightSector: handover_point := 9000
RightSector -> RightSector: endpoint := 10000
activate Train
note over of Train: <<ode>> {0, 0, 0 | p_dot=v, v_dot=a, a_dot=0 & p<terminus}
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

Translated output:
```
Train ::= ⟨terminus:=10000⟩.{0, 0, 0 | p_dot=v, v_dot=a, a_dot=0 & p<terminus, {p̅, v̅, a}}.0 || LeftSector ::= ⟨handover_point:=4000⟩.⟨endpoint:=5000⟩.p(p0).μloop[1] if p0 < handover_point then v(v0).⟨a0:=f(p0, v0, endpoint)⟩.a̅⟨a0⟩.wait(1.0).p(p0).loop[1]̅⟨⟩.0 else handover̅⟨⟩.yes().channels̅⟨p, v, a⟩.0 || RightSector ::= ⟨handover_point:=9000⟩.⟨endpoint:=10000⟩.handover().yes̅⟨⟩.channels(p1, v1, a1).p1(p0).μloop[2] if p0 < handover_point then v1(v0).⟨a0:=f(p0, v0, endpoint)⟩.a1̅⟨a0⟩.wait(1.0).p1(p0).loop[2]̅⟨⟩.0 else a1̅⟨-1⟩.0
```

Standardized output:
```
Train ::= terminus̅⟨10000⟩.{0, 0, 0 | p_dot=v, v_dot=a, a_dot=0 & p<terminus, {p̅, v̅, a}}.0
LeftSector ::= handover_point_1̅⟨4000⟩.endpoint_1̅⟨5000⟩.p(p0).loop[1]⟨⟩.0
Replication 1 ::= !loop[1]().([p0 < handover_point_1].v(v0).a0_1̅⟨f(p0, v0, endpoint_1)⟩.a̅⟨a0_1⟩.{0 | t_dot=1 & t<1.0, {}}.p(p0).loop[1]⟨⟩.0 + [(¬p0 < handover_point_1)].handover̅⟨⟩.yes().channels̅⟨p, v, a⟩.0)
RightSector ::= handover_point_2̅⟨9000⟩.endpoint_2̅⟨10000⟩.handover().yes̅⟨⟩.channels(p1, v1, a1).p1(p0).loop[2]⟨⟩.0
Replication 2 ::= !loop[2]().([p0 < handover_point_2].v1(v0).a0_2̅⟨f(p0, v0, endpoint_2)⟩.a1̅⟨a0_2⟩.{0 | t'_dot=1 & t'<1.0, {}}.p1(p0).loop[2]⟨⟩.0 + [(¬p0 < handover_point_2)].a1̅⟨-1⟩.0)
Memory0 ::= !endpoint_2(y).(endpoint_2̅⟨y⟩.endpoint_2̅⟨y⟩.0 + endpoint_2(z).endpoint_2̅⟨z⟩.0)
Memory1 ::= !handover_point_2(y).(handover_point_2̅⟨y⟩.handover_point_2̅⟨y⟩.0 + handover_point_2(z).handover_point_2̅⟨z⟩.0)
Memory2 ::= !endpoint_1(y).(endpoint_1̅⟨y⟩.endpoint_1̅⟨y⟩.0 + endpoint_1(z).endpoint_1̅⟨z⟩.0)
Memory3 ::= !handover_point_1(y).(handover_point_1̅⟨y⟩.handover_point_1̅⟨y⟩.0 + handover_point_1(z).handover_point_1̅⟨z⟩.0)
Memory4 ::= !terminus(y).(terminus̅⟨y⟩.terminus̅⟨y⟩.0 + terminus(z).terminus̅⟨z⟩.0)
System ::= (ν terminus) (ν handover_point_1) (ν endpoint_1) (ν loop[1]) (ν handover_point_2) (ν endpoint_2) (ν loop[2]) (ν a0_1) (ν t) (ν t') (ν a0_2) Train || LeftSector || Replication 1 || RightSector || Replication 2 || Memory0 || Memory1 || Memory2 || Memory3 || Memory4
```
## Development Environment

Python 3.9.18
typing_extensions   4.12.2
