# Modeling and Analysis of Cyber-Physical Systems in the Hybrid π-Calculus Using Extended Sequence Diagrams
本项目实现了一个将扩展序列图(ESD)转换为混成π演算进程(HpC)的工具。该工具可以解析特定格式的序列图描述，并将其转换为形式化的HPC进程表达式。

## 文件结构

- `hpc.py`: 定义了HPC的语法结构，包括前缀(prefix)和进程(process)类型
- `esd.py`: 实现了ESD到HpC的转换逻辑，包含各种片段类型的解析和转换
- `example.txt`: 输入文件，包含序列图描述
- `translated_output.txt`: 转换结果输出文件

## 使用方法

1. 准备输入文件`example.txt`，按照指定格式编写序列图描述
2. 运行`esd.py`主程序:
   ```bash
   python esd.py
   ```
3. 查看转换结果:
   - 控制台输出
   - 生成的`translated_output.txt`文件

## 示例

输入示例(`example.txt`):
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

转换结果:
```
Train ::= ⟨terminus:=10000⟩.{0, 0, 0 | p_dot=v, v_dot=a, a_dot=0 & p<terminus, {p̅, v̅, a}}.0 || LeftSector ::= ⟨handover_point:=4000⟩.⟨endpoint:=5000⟩.p(p0).μloop[1] if p0 < handover_point then v(v0).⟨a0:=f(p0, v0, endpoint)⟩.a̅⟨a0⟩.wait(1.0).p(p0).loop[1]̅⟨⟩.0 else handover̅⟨⟩.yes().channels̅⟨p, v, a⟩.0 || RightSector ::= ⟨handover_point:=9000⟩.⟨endpoint:=10000⟩.handover().yes̅⟨⟩.channels(p1, v1, a1).p1(p0).μloop[2] if p0 < handover_point then v1(v0).⟨a0:=f(p0, v0, endpoint)⟩.a1̅⟨a0⟩.wait(1.0).p1(p0).loop[2]̅⟨⟩.0 else a1̅⟨-1⟩.0
```

## 开发说明

Python 3.9.18
typing_extensions   4.12.2
