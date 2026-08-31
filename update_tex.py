import pandas as pd
import numpy as np
import re
import os

def parse_val(val):
    val = str(val).strip()
    if val == '' or val == 'nan' or val == 'N/A':
        return np.nan
    if '(' in val:
        val = val.split('(')[0].strip()
    try:
        return float(val)
    except:
        return np.nan

def format_val(val):
    if np.isnan(val): return "-"
    if val.is_integer():
        return str(int(val))
    return f"{val:.1f}"

def extract_float(s):
    match = re.search(r'-?\d+\.?\d*', str(s).replace(',', ''))
    if match:
        return float(match.group(0))
    return None

def calc_iqm(arr):
    if not arr: return np.nan
    arr = np.sort(arr)
    n = len(arr)
    low = int(n * 0.25)
    high = n - low
    trimmed = arr[low:high]
    return np.mean(trimmed) if len(trimmed) > 0 else np.nan

def format_table(tex_file_path):
    with open(tex_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def extract_number_str(cell_str):
        match = re.search(r'-?\d+\.?\d*', cell_str)
        if match:
            return match.group(0)
        return None

    new_lines = []
    in_table = False
    table_header_found = False

    for line in lines:
        if r'\begin{tabular}{lrrrrrrrr}' in line:
            table_header_found = True
            new_lines.append(line)
            continue
        
        if table_header_found and r'\midrule' in line:
            in_table = True
            new_lines.append(line)
            continue
        
        if in_table and r'\bottomrule' in line:
            in_table = False
            new_lines.append(line)
            continue
            
        if in_table and '&' in line:
            parts = line.split('&')
            if len(parts) >= 9:
                row_label = parts[0].strip()
                is_lower_better = "Optimality Gap" in row_label
                
                # Columns 3,5,7 are Base. Columns 4,6,8 are +ours. (0-indexed)
                for base_idx, ours_idx in [(3, 4), (5, 6), (7, 8)]:
                    base_str = parts[base_idx].strip()
                    ours_str = parts[ours_idx].replace(r'\\', '').replace('\n', '').strip()
                    
                    base_num_str = extract_number_str(base_str)
                    ours_num_str = extract_number_str(ours_str)
                    
                    if base_num_str is not None and ours_num_str is not None:
                        base_val = float(base_num_str)
                        ours_val = float(ours_num_str)
                        
                        if base_val == 0:
                            if ours_val > 0: diff_pct = float('inf')
                            elif ours_val < 0: diff_pct = float('-inf')
                            else: diff_pct = 0.0
                        else:
                            diff_pct = ((ours_val - base_val) / abs(base_val)) * 100
                            
                        # Determine color
                        if is_lower_better:
                            if diff_pct < 0: color = "blue"
                            elif diff_pct > 0: color = "red"
                            else: color = "black"
                        else:
                            if diff_pct > 0: color = "blue"
                            elif diff_pct < 0: color = "red"
                            else: color = "black"
                            
                        # Color and bold the number itself
                        if color != "black":
                            if abs(diff_pct) >= 15.0:
                                formatted_ours = f"\\textcolor{{{color}}}{{\\textbf{{{ours_num_str}}}}}"
                            else:
                                formatted_ours = f"\\textcolor{{{color}}}{{{ours_num_str}}}"
                        else:
                            formatted_ours = ours_num_str
                            
                        # Maintain newline and \\ for the last column
                        if ours_idx == 8:
                            parts[ours_idx] = f" {formatted_ours} \\\\\n"
                        else:
                            parts[ours_idx] = f" {formatted_ours} "
                
                new_line = "&".join(parts)
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(tex_file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


def main():
    excel_path = 'converted_results.xlsx'
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return
        
    df = pd.read_excel(excel_path, sheet_name='Results', index_col=[0, 1])
    df = df.reset_index()
    
    df.rename(columns={df.columns[0]: 'Game', df.columns[1]: 'Config'}, inplace=True)
    df['Game'] = df['Game'].ffill()
    
    games = df['Game'].unique()
    results = {}
    
    for game in games:
        game_df = df[df['Game'] == game]
        c1_row = game_df[game_df['Config'].astype(str) == 'Retrieval 미사용']
        c2_row = game_df[game_df['Config'].astype(str) == 'Warmup: 50000, BSR: retrieved']
        
        if c1_row.empty or c2_row.empty:
            continue
            
        c1 = c1_row.iloc[0].to_dict()
        c2 = c2_row.iloc[0].to_dict()
        
        seeds = [c for c in df.columns if str(c).strip().isdigit()]
        
        valid_seeds = []
        for s in seeds:
            v1 = parse_val(c1[s])
            v2 = parse_val(c2[s])
            if not np.isnan(v1) and not np.isnan(v2):
                valid_seeds.append(s)
                
        if valid_seeds:
            v1_mean = np.mean([parse_val(c1[s]) for s in valid_seeds])
            v2_mean = np.mean([parse_val(c2[s]) for s in valid_seeds])
            results[game] = (format_val(v1_mean), format_val(v2_mean))
            print(f"[{game}] Common seeds: {valid_seeds} -> STORM: {results[game][0]}, STORM+ours: {results[game][1]}")

    tex_path = '../iclr2027_conference.tex'
    if not os.path.exists(tex_path):
        print(f"Error: {tex_path} not found.")
        return

    with open(tex_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    hns_dict = {3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    
    # Pass 1: update data rows and calculate HNS
    for i, line in enumerate(lines):
        if line.strip().startswith(r'\#') or line.strip().startswith('Mean') or line.strip().startswith('Median') or line.strip().startswith('IQM') or line.strip().startswith('Optimality'):
            continue
            
        match = re.match(r'^([A-Za-z]+)\s*&', line)
        if match:
            game_name = match.group(1)
            if game_name == 'Game':
                continue
                
            parts = line.split('&')
            if len(parts) >= 9:
                if game_name in results:
                    parts[3] = f" {results[game_name][0]} "
                    parts[4] = f" {results[game_name][1]} "
                
                lines[i] = '&'.join(parts)
                
                rand_val = extract_float(parts[1])
                hum_val = extract_float(parts[2])
                
                if rand_val is not None and hum_val is not None and (hum_val - rand_val) != 0:
                    denominator = hum_val - rand_val
                    for col in range(3, 9):
                        score = extract_float(parts[col])
                        if score is not None:
                            hns = (score - rand_val) / denominator
                            hns_dict[col].append(hns)
                            
    # Calculate metrics
    metrics_res = {
        '#Superhuman': {},
        'Mean': {},
        'Median': {},
        'IQM': {},
        'Optimality Gap': {}
    }
    
    for col in range(3, 9):
        arr = hns_dict[col]
        if len(arr) > 0:
            metrics_res['#Superhuman'][col] = sum(1 for x in arr if x > 1.0)
            metrics_res['Mean'][col] = np.mean(arr)
            metrics_res['Median'][col] = np.median(arr)
            metrics_res['IQM'][col] = calc_iqm(arr)
            metrics_res['Optimality Gap'][col] = np.mean([max(0.0, 1.0 - x) for x in arr])
        else:
            for k in metrics_res:
                metrics_res[k][col] = None
                
    # Pass 2: Write metrics to table
    for i, line in enumerate(lines):
        metric_key = None
        if line.strip().startswith(r'\#Superhuman'):
            metric_key = '#Superhuman'
        elif line.strip().startswith('Mean'):
            metric_key = 'Mean'
        elif line.strip().startswith('Median'):
            metric_key = 'Median'
        elif line.strip().startswith('IQM'):
            metric_key = 'IQM'
        elif line.strip().startswith('Optimality Gap'):
            metric_key = 'Optimality Gap'
            
        if metric_key:
            parts = line.split('&')
            if len(parts) >= 9:
                for col in range(3, 9):
                    val = metrics_res[metric_key][col]
                    if val is not None:
                        if metric_key == '#Superhuman':
                            formatted = f" {int(val)} "
                        else:
                            formatted = f" {val:.3f} "
                    else:
                        formatted = " - "
                        
                    if col == 8:
                        parts[col] = formatted + " \\\\\n"
                    else:
                        parts[col] = formatted + " "
                lines[i] = '&'.join(parts)

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
        
    print(f"\nSuccessfully updated {tex_path} with matching seed averages and calculated metrics!")
    
    format_table(tex_path)
    print("Table formatting complete! Only the scores are colored/bolded now.")

if __name__ == '__main__':
    main()
