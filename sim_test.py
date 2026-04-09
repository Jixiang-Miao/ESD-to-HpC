import os
import simulator as sim
from esd import ESD, parse_example

# Batch test configuration: input file | role list | case number
TEST_CASES = [
    ("example\\example1.txt", ["Alice", "Bob"], 1),
    ("example\\example2.txt", ["Alice", "Bob"], 2),
    ("example\\example3.txt", ["Alice", "Bob"], 3),
    ("example\\example4.txt", ["Alice", "Bob"], 4),
    ("example\\example5.txt", ["Alice"], 5),
    ("example\\example6.txt", ["Alice"], 6),
    ("example\\example7.txt", ["Alice", "Bob"], 7),
    ("example\\example8.txt", ["Alice", "Bob"], 8),
    ("example\\example9.txt", ["Alice", "Bob", "Charlie"], 9),
    ("example\\example10.txt", ["Alice", "Bob"], 10),
    ("example\\example11.txt", ["Alice", "Bob"], 11),
    ("example\\example12.txt", ["Alice", "Bob"], 12),
    ("example\\example13.txt", ["Alice", "Bob"], 13),
    ("example\\example14.txt", ["Alice", "Bob"], 14),
    ("example\\example15.txt", ["Alice", "Bob"], 15),
    ("example\\example16.txt", ["Alice", "Bob"], 16),
    ("example\\example17.txt", ["Alice", "Bob"], 17),
    ("example\\example18.txt", ["Alice", "Bob"], 18),
    ("example\\example19.txt", ["Alice", "Bob"], 19),
    ("example\\example20.txt", ["Alice", "Bob"], 20),
    ("example\\example21.txt", ["Alice", "Bob", "Charlie"], 21),
    ("example\\example22.txt", ["Alice", "Bob", "Charlie"], 22),
    ("example\\example23.txt", ["Alice", "Bob"], 23),
    ("example\\example24.txt", ["Alice", "Bob", "Charlie"], 24),
    ("example\\example25.txt", ["Alice", "Bob", "Charlie", "David"], 25),
    ("example\\example26.txt", ["Alice", "Bob"], 26),
    ("example\\example27.txt", ["Alice", "Bob"], 27),
    ("example\\example28.txt", ["Alice", "Bob"], 28),
    ("example\\example29.txt", ["Alice", "Bob"], 29),
    ("example\\example30.txt", ["Alice", "Bob"], 30),
    ("example\\example31.txt", ["Alice", "Bob"], 31),
    ("example\\example32.txt", ["Alice", "Bob"], 32),
    ("example\\example33.txt", ["Alice", "Bob"], 33),
    ("example\\example34.txt", ["Alice", "Bob"], 34),
    ("example\\example35.txt", ["Alice", "Bob", "Charlie"], 35),
    ("example\\example36.txt", ["Alice", "Bob", "Charlie"], 36),
    ("example\\example37.txt", ["Alice", "Bob"], 37),
    ("example\\example38.txt", ["Alice", "Bob"], 38),
    ("example\\example39.txt", ["Alice", "Bob"], 39),
    ("example\\example40.txt", ["Alice", "Bob"], 40),
    ("example\\example41.txt", ["Alice", "Bob"], 41),
    ("example\\example42.txt", ["Alice", "Bob", "Charlie"], 42)
]

def run_single_case(input_file, roles, case_num):
    case_prefix = f"example\\example{case_num}"
    translate_file = f"{case_prefix}_translated_output.txt"
    standard_file = f"{case_prefix}_standardized_output.txt"
    events_file = f"{case_prefix}_simulation_events.txt"
    trace_file = f"{case_prefix}_trace.txt"
    step_file = f"{case_prefix}_steps.txt"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            input_content = f.read()

        root = parse_example(input_content)
        esd = ESD(root, roles)
        translated_result = esd.translate()

        with open(translate_file, "w", encoding="utf-8") as f:
            f.write(str(translated_result))

        standardized_result = sim.to_standard_form(translated_result)
        with open(standard_file, "w", encoding="utf-8") as f:
            f.write(str(standardized_result))

        sim.EVENTS_FILE = events_file
        sim.TRACE_FILE = trace_file
        sim.STEP_FILE = step_file

        sim.simulate_block(standardized_result, max_steps=10000, init_state={"roles": roles})
        sim.flush_logs()

        print(f"✅ Case {case_num} completed: Generated translation, standardized and simulation log files")
    except Exception as e:
        print(f"❌ Case {case_num} failed: {e}")

if __name__ == "__main__":
    os.makedirs("example", exist_ok=True)

    for input_file, roles, case_num in TEST_CASES:
        if os.path.exists(input_file):
            run_single_case(input_file, roles, case_num)
        else:
            print(f"⚠️  Case {case_num} skipped: Input file {input_file} does not exist")

    print("\n📋 Batch simulation test execution completed")