# Modeling and Analysis of Cyber-Physical Systems in the Hybrid π-Calculus Using Extended Sequence Diagrams
This project implements a tool that converts Extended Sequence Diagram (ESD) into Hybrid π-Calculus (HpC) processes. The tool parses ESD written in a specific format and translates them into formal HpC process expressions.

## File Structure

- `hpc.py`: Defines the syntax of HpC, including prefixes and process types.
- `esd.py`: Contains the logic for converting ESDs to HpC, including parsing and transformation of different fragment types.
- `example.txt`: Input file containing the sequence diagram description.
- `translated_output.txt`: Output file containing the translated result.

## Usage

1. Prepare the input file 'example.txt' following the specified sequence diagram format.
2. Run the main script 'esd.py':
   ```bash
   python esd.py
   ```
3. View the translation results:
   - Console output
   - The generated`translated_output.txt` file

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

## Development Environment

Python 3.9.18
typing_extensions   4.12.2
