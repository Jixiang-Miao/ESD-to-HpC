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
    ("example\\example9.txt", ["Alice", "Bob", "Charlie", "David"], 9),
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