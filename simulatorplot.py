import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

def read_trace_file_automatically(filepath):
    """Parse trace file in segments: each time encountering a header line (containing 'global_time' and separated by commas)
    start a new block, read subsequent data until the next header; parse each block separately into a DataFrame,
    finally merge (outer join) and convert numeric columns to numeric types."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    blocks = []
    current_header = None
    current_data = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue

        if 'global_time' in line and ',' in line:
            if current_header is not None and current_data:
                blocks.append((current_header, current_data.copy()))
            current_header = line
            current_data = []
        else:
            if current_header is None:
                continue
            current_data.append(line)

    if current_header is not None and current_data:
        blocks.append((current_header, current_data.copy()))

    dfs = []
    for header, data_lines in blocks:
        data_str = header + '\n' + '\n'.join(data_lines)
        try:
            df_block = pd.read_csv(StringIO(data_str))
            dfs.append(df_block)
        except Exception as e:
            print(f"skipping block that could not be parsed (error: {e})")

    if not dfs:
        return None, []

    # merge all blocks (even if columns are not identical)
    df = pd.concat(dfs, ignore_index=True, sort=False)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # construct variable names: ensure global_time is first (if present)
    cols = list(df.columns)
    if 'global_time' in cols:
        cols.remove('global_time')
        variable_names = ['global_time'] + cols
    else:
        variable_names = cols

    return df, variable_names

def plot_continuous_evolution_auto(filepath):
    """plot continuous evolution of variables in a continuous process (supporting multiple headers in the file)"""

    df, variable_names = read_trace_file_automatically(filepath)

    if df is None or len(variable_names) < 2:
        print("skipping block that could not be parsed (error: {e})")
        return
    
    # if 'p' in df.columns:
    #     df = df.drop(columns=['p'])

    print(f"data shape: {df.shape}")
    print(f"available variables: {variable_names}")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    line_styles = ['-', '--', '-.', ':']

    time_var = 'global_time'
    data_vars = [v for v in variable_names[1:] if v in df.columns]

    for i, var in enumerate(data_vars):
        if df[var].notna().sum() == 0:
            continue
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        plt.plot(df[time_var], df[var],
                 color=color,
                 linewidth=2,
                 label=f'{var}',
                 linestyle=line_style)

    plt.xlabel(f'{time_var}', fontsize=12)
    plt.ylabel('Variable Value', fontsize=12)
    plt.title('Continuous Process Variable Evolution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.xlim(df[time_var].min(), df[time_var].max())
    plt.tight_layout()
    plt.savefig("trace.png", dpi=1200, bbox_inches='tight')
    plt.show()

    return df, variable_names

if __name__ == "__main__":
    df, vars = plot_continuous_evolution_auto("trace.txt")
    if df is not None:
        print("\nData Statistics:")
        print(df.describe())