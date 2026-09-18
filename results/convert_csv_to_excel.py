import pandas as pd
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Side, Font
from openpyxl.utils import get_column_letter

# (Random, Human, STORM 논문) 고정 참조 점수.
# Random/Human: iclr2027_conference.tex의 tab:main_performance (2026-09-17).
# STORM: 첨부된 논문 표의 STORM (ours) 열. 실행 시 원본 문서를 읽지 않습니다.
REFERENCE_SCORES = {
    'Alien': (227.8, 7127.7, 984),
    'Amidar': (5.8, 1719.5, 205),
    'Assault': (222.4, 742.0, 801),
    'Asterix': (210.0, 8503.3, 1028),
    'BankHeist': (14.2, 753.1, 641),
    'BattleZone': (2360.0, 37187.5, 13540),
    'Boxing': (0.1, 12.1, 80),
    'Breakout': (1.7, 30.5, 16),
    'ChopperCommand': (811.0, 7387.8, 1888),
    'CrazyClimber': (10780.5, 35829.4, 66776),
    'DemonAttack': (152.1, 1971.0, 165),
    'Freeway': (0.0, 29.6, 34),
    'Frostbite': (65.2, 4334.7, 1316),
    'Gopher': (257.6, 2412.5, 8240),
    'Hero': (1027.0, 30826.4, 11044),
    'Jamesbond': (29.0, 302.8, 509),
    'Kangaroo': (52.0, 3035.0, 4208),
    'Krull': (1598.0, 2665.5, 8413),
    'KungFuMaster': (258.5, 22736.3, 26182),
    'MsPacman': (307.3, 6951.6, 2673),
    'Pong': (-20.7, 14.6, 11),
    'PrivateEye': (24.9, 69571.3, 7781),
    'Qbert': (163.9, 13455.0, 4522),
    'RoadRunner': (11.5, 7845.0, 17564),
    'Seaquest': (68.4, 42054.7, 525),
    'UpNDown': (533.4, 11693.2, 7985),
}

PAIRED_MEAN_COLUMN = 'Mean (공통 시드)'
SCORE_DELTA_COLUMN = 'Δ Score (행별 비교)'
HNS_DELTA_COLUMN = 'Δ HNS (행별 비교)'


def parse_score(value):
    """update_tex.py의 parse_val과 동일하게 셀의 첫 번째 점수만 사용."""
    value = str(value).strip()
    if value.lower() in {'', 'nan', 'n/a', 'na', 'none', 'running'}:
        return float('nan')
    value = value.split(',')[0].split('(')[0].strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def add_paired_baseline_mean(pivot_df):
    """공통 시드 평균과 논문/target 16 행의 baseline 대비 비교를 추가."""
    seed_columns = [column for column in pivot_df.columns if str(column).strip().isdigit()]
    columns = list(pivot_df.columns)
    if 6020 not in columns:
        columns.append(6020)
    columns += [
        '   ', PAIRED_MEAN_COLUMN, SCORE_DELTA_COLUMN, HNS_DELTA_COLUMN,
    ]
    pivot_df = pivot_df.reindex(columns=columns, fill_value='')

    for game in pivot_df.index.get_level_values('Game').unique():
        baseline_key = (game, 'Retrieval 미사용')
        target_key = (game, 'target: 16 (anchor 미설정)')
        if baseline_key not in pivot_df.index or target_key not in pivot_df.index:
            continue
        baseline = pivot_df.loc[baseline_key, seed_columns].map(parse_score)
        target = pivot_df.loc[target_key, seed_columns].map(parse_score)
        valid = baseline.notna() & target.notna()
        if valid.any():
            baseline_mean = baseline[valid].mean()
            target_mean = target[valid].mean()
            target_delta = target_mean - baseline_mean
            pivot_df.at[baseline_key, PAIRED_MEAN_COLUMN] = baseline_mean
            pivot_df.at[target_key, PAIRED_MEAN_COLUMN] = target_mean
            pivot_df.at[target_key, SCORE_DELTA_COLUMN] = abs(target_delta)
            if game in REFERENCE_SCORES:
                random_score, human_score, paper_score = REFERENCE_SCORES[game]
                paper_key = (game, f'STORM (논문): {paper_score}')
                score_delta = baseline_mean - paper_score
                if paper_key in pivot_df.index:
                    pivot_df.at[paper_key, SCORE_DELTA_COLUMN] = score_delta
                if human_score != random_score:
                    # HNS는 Random=0, Human=1 기준의 정규화 점수이며 백분율이 아닙니다.
                    pivot_df.at[target_key, HNS_DELTA_COLUMN] = (
                        target_delta / (human_score - random_score)
                    )
                    if paper_key in pivot_df.index:
                        pivot_df.at[paper_key, HNS_DELTA_COLUMN] = (
                            score_delta / (human_score - random_score)
                        )
    return pivot_df


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
        created_at_utc = pd.to_datetime(row.get('Created At', '2000-01-01T00:00:00Z'), utc=True)
        
        # ---------------------------------------------------------
        # 날짜 필터링: EMA 기법이 최신 코드에 적용된 시점 이후의 실험만 포함
        # 기준 시점: 2026-08-24 11:05:43 UTC
        #
        # [AGENT INSTRUCTION] 
        # Retrieval 미사용(ret_enable == False) 설정은 이 날짜 필터링을 무시하고
        # 모든 시점의 데이터를 가져오며, 중복 시 최신 점수를 기록하도록 항상 이 로직을 유지해야 합니다.
        # 향후 이 코드를 수정할 계획을 세울 때, AI 에이전트는 반드시 사용자에게 
        # "Retrieval 미사용은 예전 데이터를 유지하는 로직이 적용되어 있다"는 사실을 사전에 알려야 합니다.
        # ---------------------------------------------------------
        ema_applied_date = pd.to_datetime("2026-08-24T11:05:43Z", utc=True)
        if ret_enable and created_at_utc < ema_applied_date:
            continue
        
        # Safely parse warmup_steps to int if possible
        try:
            w_float = float(row['Warmup Steps'])
            if not pd.isna(w_float):
                warmup_steps = int(w_float)
            else:
                warmup_steps = 'N/A'
        except (ValueError, TypeError):
            warmup_steps = 'N/A'

        try:
            cw_float = float(row.get('Calculated Warmup Steps', row['Warmup Steps']))
            if not pd.isna(cw_float):
                calculated_warmup_steps = int(cw_float)
            else:
                calculated_warmup_steps = 'N/A'
        except (ValueError, TypeError, KeyError):
            calculated_warmup_steps = 'N/A'

        # ---------------------------------------------------------
        # 실험 필터링 및 표기법 변경 로직
        # 1. Retrieval을 사용하지 않은 실험은 'Retrieval 미사용'으로 표기
        # 2. Retrieval을 사용한 실험 중 특정 조건(Warmup 50000, BSR retrieved, Z score 3.5)만 필터링
        # 3. target 1 또는 16이고 anchor 미설정인 실험만 유지 (기존대로 0.0625도 미설정 취급)
        # ---------------------------------------------------------
        if not ret_enable:
            config = 'Retrieval 미사용'
        else:
            bsr = row.get('Batch Size Reduction', 'N/A')
            if pd.isna(bsr) or str(bsr).strip() == '' or str(bsr) == 'nan':
                bsr = 'N/A'

            z_score = row.get('Z Score Threshold', 'N/A')
            try:
                z_score_float = float(z_score)
            except (ValueError, TypeError):
                z_score_float = None

            if warmup_steps == 50000 and str(bsr) == 'retrieved' and z_score_float == 3.5:
                r_target = row.get('Retrieval Target', 'N/A')
                try:
                    r_target = float(r_target)
                    if r_target not in (1, 16):
                        continue
                except (ValueError, TypeError):
                    continue
                
                a_weight = row.get('Anchor Weight', 'N/A')
                try:
                    anchor_is_unset = float(a_weight) == 0.0625
                except (ValueError, TypeError):
                    anchor_is_unset = False

                if (pd.isna(a_weight) or str(a_weight).strip() == '' or
                        str(a_weight) == 'nan' or str(a_weight) == 'N/A' or
                        anchor_is_unset):
                    config = f'target: {int(r_target)} (anchor 미설정)'
                else:
                    continue
            else:
                # 위 경우에 해당하지 않는 실험은 제외
                continue
                
        eval_return = row['Eval Return']
        h_bits = row.get('Hash Bits', 'N/A')
        if pd.isna(h_bits) or str(h_bits).strip() == '' or str(h_bits) == 'nan':
            h_bits = 'N/A'

        data.append({
            'Game': game,
            'Config': config,
            'Seed': seed,
            'Eval Return': eval_return,
            'Warmup Steps': calculated_warmup_steps,
            'Hash Bits': h_bits,
            'Created At': created_at_utc
        })

    # 강제로 Frostbite - Retrieval 미사용의 누락된 시드 추가
    data.append({
        'Game': 'Frostbite',
        'Config': 'Retrieval 미사용',
        'Seed': 10,
        'Eval Return': '2068',
        'Warmup Steps': 'N/A',
        'Hash Bits': 'N/A',
        'Created At': pd.to_datetime("2026-08-11T04:36:29Z", utc=True)
    })
    data.append({
        'Game': 'Frostbite',
        'Config': 'Retrieval 미사용',
        'Seed': 3710,
        'Eval Return': '1904',
        'Warmup Steps': 'N/A',
        'Hash Bits': 'N/A',
        'Created At': pd.to_datetime("2026-08-11T04:36:35Z", utc=True)
    })

    parsed_df = pd.DataFrame(data)

    # 1. 성능이 N/A인 것은 무시
    parsed_df['Eval Return'] = parsed_df['Eval Return'].astype(str)
    parsed_df = parsed_df[
        (parsed_df['Eval Return'] != 'N/A') & 
        (parsed_df['Eval Return'] != 'nan') &
        (parsed_df['Eval Return'].str.strip() != '')
    ]
    
    # 2. 모든 실험 결과 표시 (중복 런 포함 모두 나열하기 위해 중복 제거 로직 삭제)
    # 먼저 Created At 기준으로 내림차순 정렬 (최신이 위로 오도록)
    parsed_df = parsed_df.sort_values(by='Created At', ascending=False)
    
    # 3. 동일한 시드에서 여러 결과가 있을 경우, 셀 하나에 여러 줄로 나열
    def aggregate_cell(group):
        def format_single_row(r):
            val = r['Eval Return']
            try:
                val = f"{float(val):.2f}"
            except ValueError:
                pass
                
            # 단일 결과여도 50000이 아니면 명시
            warmup = r['Warmup Steps']
            try:
                w_val = int(float(warmup))
                if w_val != 50000:
                    val = f"{val} (w: {w_val})"
            except (ValueError, TypeError):
                pass
                
            return str(val)

        if len(group) == 1:
            return format_single_row(group.iloc[0])
        else:
            # Hash Bits가 10인 실험이 존재한다면, 해당 실험들만 남깁니다.
            is_hb_10 = group['Hash Bits'].apply(lambda x: str(x).strip() in ['10', '10.0'])
            if is_hb_10.any():
                group = group[is_hb_10]
                
            if len(group) == 1:
                return format_single_row(group.iloc[0])

            items = []
            has_diff_hash = group['Hash Bits'].nunique() > 1
            has_diff_warmup = group['Warmup Steps'].nunique() > 1
            for _, r in group.iterrows():
                val = r['Eval Return']
                try:
                    val = f"{float(val):.2f}"
                except ValueError:
                    pass
                
                extras = []
                if has_diff_hash:
                    extras.append(f"hb: {r['Hash Bits']}")
                
                warmup = r['Warmup Steps']
                # 값이 다를 경우 구분을 위해 무조건 덧붙여 표기
                if has_diff_warmup or (str(warmup) != '50000' and str(warmup) != 'N/A'):
                    extras.append(f"w: {warmup}")
                
                if extras:
                    items.append(f"{val} ({', '.join(extras)})")
                else:
                    items.append(f"{val}")
            return ", ".join(items)

    agg_df = parsed_df.groupby(['Game', 'Config', 'Seed']).apply(aggregate_cell, include_groups=False).reset_index(name='Final Eval Return')
    
    # Pivot table
    pivot_df = agg_df.pivot_table(
        index=['Game', 'Config'],
        columns='Seed',
        values='Final Eval Return',
        aggfunc='first'
    )

    # Target column order
    target_columns = [1, 2, 10, 710, 3710, ' ', 2000, 2010, '  ', 5090]
    extra_seeds = [c for c in pivot_df.columns if c not in [1, 2, 10, 710, 3710, 2000, 2010, 5090]]
    full_columns = target_columns + extra_seeds
    
    pivot_df = pivot_df.reindex(columns=full_columns)
    pivot_df = pivot_df.fillna('')

    # 논문 점수는 시드별 실험값과 구분하여 Config 열의 별도 첫 행에 표시합니다.
    reference_index = pd.MultiIndex.from_tuples(
        [(game, f'STORM (논문): {REFERENCE_SCORES[game][2]}')
         for game in pivot_df.index.get_level_values('Game').unique()
         if game in REFERENCE_SCORES],
        names=['Game', 'Config'],
    )
    reference_df = pd.DataFrame('', index=reference_index, columns=pivot_df.columns)
    pivot_df = pd.concat([pivot_df, reference_df])

    # Sorting Index
    base_order = [
        'STORM (논문)',
        'Retrieval 미사용',
        'target'
    ]

    def get_sort_key(config_str):
        for i, base in enumerate(base_order):
            if config_str.startswith(base):
                return i
        return len(base_order)

    # Sort the multi-index: alphabetical by Game, then by specified Config order
    sorted_index = sorted(pivot_df.index, key=lambda x: (x[0], get_sort_key(x[1])))
    pivot_df = pivot_df.reindex(sorted_index)
    pivot_df = add_paired_baseline_mean(pivot_df)

    output_path = 'converted_results.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pivot_df.to_excel(writer, sheet_name='Results')
        
        workbook = writer.book
        worksheet = writer.sheets['Results']

        mean_col = pivot_df.columns.get_loc(PAIRED_MEAN_COLUMN) + 3
        worksheet.column_dimensions[get_column_letter(mean_col - 1)].width = 3
        for offset, (width, number_format) in enumerate([
            (30, '0.00'),
            (46, '+0.00;-0.00;0.00'),
            (32, '+0.0000;-0.0000;0.0000'),
        ]):
            column = mean_col + offset
            worksheet.column_dimensions[get_column_letter(column)].width = width
            for row in worksheet.iter_rows(min_row=2, min_col=column, max_col=column):
                row[0].number_format = number_format

        worksheet.cell(row=1, column=mean_col).comment = Comment(
            'Retrieval 미사용과 target: 16 양쪽에 유효한 점수가 있는 공통 시드만 사용합니다. '
            '각 시드에 여러 결과가 있으면 첫 번째 점수를 사용합니다.',
            'STORM',
        )
        worksheet.cell(row=1, column=mean_col + 1).comment = Comment(
            'STORM (논문) 행: Retrieval 미사용의 공통 시드 평균 − STORM 논문 점수.\n'
            'target: 16 행: |target: 16의 공통 시드 평균 − Retrieval 미사용의 공통 시드 평균|.',
            'STORM',
        )
        worksheet.cell(row=1, column=mean_col + 2).comment = Comment(
            'STORM (논문) 행: (Retrieval 미사용 평균 − STORM 논문 점수) / (Human − Random).\n'
            'target: 16 행: (target: 16 평균 − Retrieval 미사용 평균) / (Human − Random).\n'
            '모든 평균은 공통 시드 기준이며 HNS 차이는 부호를 유지합니다. 백분율이 아닙니다.',
            'STORM',
        )

        start_row = worksheet.max_row - len(pivot_df) + 1
        game_row_counts = pivot_df.groupby(level='Game').size()
        for row_idx, (game, config) in enumerate(pivot_df.index):
            excel_row = row_idx + start_row
            if config == 'target: 16 (anchor 미설정)':
                worksheet.cell(row=excel_row, column=mean_col + 1).number_format = '0.00'
            # 병합된 게임 셀의 세 줄이 모두 보이도록 높이와 줄바꿈 설정.
            worksheet.row_dimensions[excel_row].height = max(24, 72 / game_row_counts[game])
            if row_idx == 0 or game != pivot_df.index[row_idx - 1][0]:
                if game in REFERENCE_SCORES:
                    random_score, human_score, _ = REFERENCE_SCORES[game]
                    cell = worksheet.cell(row=excel_row, column=1)
                    cell.value = f'{game}\nRandom: {random_score:.1f}\nHuman: {human_score:.1f}'
                    cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
        
        # 1. 모든 셀 폰트 크기 16으로 설정
        for row in worksheet.iter_rows():
            for cell in row:
                if cell.font:
                    cell.font = Font(name=cell.font.name, size=16, bold=cell.font.bold, italic=cell.font.italic, color=cell.font.color)
                else:
                    cell.font = Font(size=16)
                    
        # 2. A, B열 내용에 맞게 열 너비 자동 맞춤
        for col_letter, col_idx in [('A', 1), ('B', 2)]:
            max_len = 0
            for row in worksheet.iter_rows(min_col=col_idx, max_col=col_idx):
                for cell in row:
                    if cell.value:
                        # 줄바꿈이 있는 경우 가장 긴 줄 기준
                        lines = str(cell.value).split('\n')
                        for line in lines:
                            max_len = max(max_len, len(line))
                            
            # 폰트 사이즈 16이므로, 일반(11)보다 크기 때문에 가중치(약 1.7배)를 줍니다.
            worksheet.column_dimensions[col_letter].width = (max_len * 1.7) + 2

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
