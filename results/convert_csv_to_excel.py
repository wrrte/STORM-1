import pandas as pd
import numpy as np
from openpyxl.styles import Border, Side

def main():
    # Load CSV
    df = pd.read_csv('wandb_runs_classification.csv')

    # Data extraction
    data = []
    for idx, row in df.iterrows():
        run_name = str(row['Run Name'])
        parts = run_name.split('_')
        game = parts[0]
        
        # Extract seed
        seed = str(row['Seed'])
        if seed.endswith('.0'):
            seed = seed[:-2]
        if seed == 'N/A' or seed == 'nan' or seed == 'None':
            # Try to extract from Run Name (e.g., Amidar_f05wv0rj_2000_X)
            if len(parts) >= 3 and parts[2].isdigit():
                seed = parts[2]
            elif len(parts) == 4 and parts[3].isdigit():
                seed = parts[3]
            
        if not str(seed).isdigit():
            continue # Ignore if seed cannot be found
            
        seed = int(seed)
        
        ret_enable = str(row['Retrieval Enable']).strip().lower() in ['true', '1', 't']
        
        if not ret_enable:
            config = 'retrieval false'
        else:
            logic = str(row['Logic']).strip()
            if 'without max' in logic:
                config = 'retrieved_count (without max)'
            elif 'with max' in logic:
                config = 'retrieved_count (with max)'
            elif 'num_valid_anchors' in logic:
                config = 'num_valid_anchors'
            elif 'imagine_batch_size' in logic:
                config = 'imagine_batch_size (No retrieval deduction)'
            else:
                config = logic
                
        eval_return = row['Eval Return']
        
        # Safely parse warmup_steps to int if possible
        try:
            w_float = float(row['Warmup Steps'])
            if not pd.isna(w_float):
                warmup_steps = int(w_float)
            else:
                warmup_steps = 'N/A'
        except (ValueError, TypeError):
            warmup_steps = 'N/A'
        
        data.append({
            'Game': game,
            'Config': config,
            'Seed': seed,
            'Eval Return': eval_return,
            'Warmup Steps': warmup_steps,
            'Retrieval Enable': ret_enable
        })

    parsed_df = pd.DataFrame(data)

    # Resolve overlapping runs (same Game, Config, Seed) by taking the highest Eval Return
    parsed_df['Numeric Return'] = pd.to_numeric(parsed_df['Eval Return'], errors='coerce')
    parsed_df = parsed_df.sort_values(by=['Game', 'Config', 'Seed', 'Numeric Return'], ascending=[True, True, True, False])
    
    duplicates = parsed_df[parsed_df.duplicated(subset=['Game', 'Config', 'Seed'], keep=False)]
    if not duplicates.empty:
        print("WARNING: Found multiple rewards for the same Game, Config, and Seed. Selecting the highest reward:")
        for name, group in duplicates.groupby(['Game', 'Config', 'Seed']):
            returns = group['Eval Return'].tolist()
            warmups = group['Warmup Steps'].tolist()
            print(f"  - Game: {name[0]}, Config: {name[1]}, Seed: {name[2]}")
            print(f"    Available: Returns {returns} with Warmups {warmups} -> Chosen: Return {returns[0]} (Warmup {warmups[0]})")
            
    parsed_df = parsed_df.drop_duplicates(subset=['Game', 'Config', 'Seed'], keep='first')
    parsed_df = parsed_df.reset_index(drop=True)

    # Process warmup steps per (Game, Config)
    final_configs = []
    final_evals = []

    for idx, row in parsed_df.iterrows():
        game = row['Game']
        config = row['Config']
        ret_enable = row['Retrieval Enable']
        eval_return = row['Eval Return']
        warmup = row['Warmup Steps']
        
        # Get all runs for this Game & Config
        group = parsed_df[(parsed_df['Game'] == game) & (parsed_df['Config'] == config)]
        
        final_config = config
        final_eval = str(eval_return) if pd.notna(eval_return) and str(eval_return) != 'nan' else 'N/A'
        
        if ret_enable:
            # Collect valid integer warmups
            warmups = [w for w in group['Warmup Steps'] if isinstance(w, int)]
            
            if len(warmups) > 0:
                unique_warmups = set(warmups)
                if len(unique_warmups) == 1:
                    w = list(unique_warmups)[0]
                    if w != 20000:
                        # Condition 1: all seeds share the same warmup != 20000
                        final_config = f"{config} ({w})"
                else:
                    # Condition 2: varying warmups, annotate inline if != 20000
                    if isinstance(warmup, int):
                        if warmup != 20000 and final_eval != 'N/A':
                            final_eval = f"{final_eval} ({warmup})"

        final_configs.append(final_config)
        final_evals.append(final_eval)

    parsed_df['Final Config'] = final_configs
    parsed_df['Final Eval Return'] = final_evals

    # Pivot table
    pivot_df = parsed_df.pivot_table(
        index=['Game', 'Final Config'],
        columns='Seed',
        values='Final Eval Return',
        aggfunc='first' # First value in case of multiple runs with same config and seed
    )

    # Target column order
    target_columns = [1, 2, 10, 710, 3710, ' ', 2000, 2010, '  ', 5090]
    extra_seeds = [c for c in pivot_df.columns if c not in [1, 2, 10, 710, 3710, 2000, 2010, 5090]]
    full_columns = target_columns + extra_seeds
    
    pivot_df = pivot_df.reindex(columns=full_columns)
    pivot_df = pivot_df.fillna('')

    # Sorting Index
    base_order = [
        'retrieval false',
        'retrieved_count (without max)',
        'num_valid_anchors',
        'imagine_batch_size (No retrieval deduction)'
    ]

    def get_sort_key(config_str):
        for i, base in enumerate(base_order):
            if config_str.startswith(base):
                return i
        return len(base_order)

    # Sort the multi-index: alphabetical by Game, then by specified Config order
    sorted_index = sorted(pivot_df.index, key=lambda x: (x[0], get_sort_key(x[1]), x[1]))
    pivot_df = pivot_df.reindex(sorted_index)

    output_path = 'converted_results.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pivot_df.to_excel(writer, sheet_name='Results')
        
        workbook = writer.book
        worksheet = writer.sheets['Results']
        
        # medium border for visibility between games
        top_side = Side(border_style="medium", color="000000")
        
        # Calculate start_row dynamically
        start_row = worksheet.max_row - len(pivot_df) + 1
        
        for row_idx, (game, config) in enumerate(pivot_df.index):
            excel_row = row_idx + start_row
            
            # Draw line above the first row of each new game
            if row_idx == 0 or pivot_df.index[row_idx][0] != pivot_df.index[row_idx-1][0]:
                for col_idx in range(1, len(pivot_df.columns) + 3):
                    cell = worksheet.cell(row=excel_row, column=col_idx)
                    cell.border = Border(
                        top=top_side,
                        left=cell.border.left,
                        right=cell.border.right,
                        bottom=cell.border.bottom
                    )

    print(f"Successfully saved to {output_path}")

if __name__ == '__main__':
    main()
