import json
import pandas as pd
import numpy as np
from openpyxl.styles import Border, Side

def calculate_warmup_steps(steps, rewards):
    """
    이 함수를 수정하여 원하는 warmup_steps 도출 공식을 테스트하세요!
    
    Args:
        steps (list): 스텝(x축) 배열 (예: [1000, 2000, 3000, ...])
        rewards (list): 보상(y축) 배열 (예: [10.5, 12.1, 15.0, ...])
        
    Returns:
        int: 공식에 의해 도출된 추정 warmup_steps 값
    """
    if not steps or not rewards:
        return 20000 # 베이스라인 데이터가 없을 경우 기본값 반환
        
    # Pandas를 이용해 분석하기 편하도록 변환
    s_steps = pd.Series(steps)
    s_rewards = pd.Series(rewards)
    
    # -------------------------------------------------------------
    # [여기에 튜닝할 공식을 작성하세요]
    #
    # 예시 공식 1: 이동평균(Window=5)이 특정 점수(예: 100)를 넘는 첫 시점
    # 예시 공식 2: 보상 증가율(기울기)이 특정 수치 이하로 떨어지는(플래토) 시점
    # 
    # 현재는 간단한 Dummy 로직(보상 그래프의 중간 지점 스텝 반환)이 들어있습니다.
    # -------------------------------------------------------------
    
    # 노이즈(들쭉날쭉함)를 약간 잡아주기 위해 짧은 윈도우로 스무딩(이동평균) 적용
    smoothed = s_rewards.rolling(window=5, min_periods=1).mean()
    
    # 1. 과거 데이터(예: 최근 20개 스텝)를 바탕으로 '평상시의 평균'과 '들쭉날쭉한 정도(표준편차)'를 계산
    window_size = 20
    rolling_mean = smoothed.rolling(window=window_size, min_periods=5).mean()
    rolling_std = smoothed.rolling(window=window_size, min_periods=5).std()
    
    # 2. 볼린저 밴드 상단(Upper Band) 계산
    # '평상시 평균' + '평상시 들쭉날쭉함(std)' x N배 
    # N=3 이면, 99.7%의 평범한 튀어오름을 전부 포괄하는 상한선(저항선)을 긋는 것입니다.
    # 이 선을 뚫었다는 것은 "평소와 다른 비정상적인 상승폭"이라는 뜻이 됩니다.
    threshold_multiplier = 2.6
    upper_band = rolling_mean + (threshold_multiplier * rolling_std)
    upper_band = upper_band.fillna(float('inf')) # 극초반 데이터 부족 구간 방지
    
    # 3. 스무딩된 보상이 이 상한선을 뚫고 올라간 시점(Index)들 찾기
    breakout_indices = s_rewards.index[smoothed > upper_band].tolist()
    
    if breakout_indices:
        # 최초로 상한선을 뚫고 폭발적으로 상승한 시점의 스텝 반환
        first_breakout_idx = breakout_indices[0]
        calculated_step = steps[first_breakout_idx]
    else:
        # 끝까지 그런 폭발적 상승이 없었다면 기본값 반환
        calculated_step = 20000

    return calculated_step

def main():
    print("Loading data from wandb_extracted_data.json...")
    with open('wandb_extracted_data.json', 'r', encoding='utf-8') as f:
        runs = json.load(f)
        
    # 1. Game, Seed 별로 'retrieval false' 런의 시계열 데이터를 딕셔너리로 맵핑
    # 구조: baseline_data[game][seed] = {"steps": [...], "rewards": [...]}
    baseline_data = {}
    for run in runs:
        if run['config'] == 'retrieval false':
            game = run['game']
            seed = run['seed']
            if seed is None:
                continue
            
            if game not in baseline_data:
                baseline_data[game] = {}
                
            baseline_data[game][seed] = {
                "steps": run['reward_history']['step'],
                "rewards": run['reward_history']['reward']
            }

    # 2. 모든 런에 대해 셀에 들어갈 텍스트 계산
    data_for_df = []
    
    for run in runs:
        game = run['game']
        seed = run['seed']
        config = run['config']
        eval_return = run['eval_return']
        warmup = run['warmup_steps']
        ret_enable = run['retrieval_enable']
        
        if seed is None:
            continue
            
        # 해당 Game, Seed의 베이스라인 데이터 찾기
        base_steps = []
        base_rewards = []
        if game in baseline_data and seed in baseline_data[game]:
            base_steps = baseline_data[game][seed]['steps']
            base_rewards = baseline_data[game][seed]['rewards']
            
        # 공식 적용! (항상 retrieval false의 그래프 데이터 전달)
        calc_warmup = calculate_warmup_steps(base_steps, base_rewards)
        
        # 셀 텍스트 조합
        # 1. Eval Return
        cell_text = str(eval_return) if eval_return != 'N/A' else 'N/A'
        
        # 2. Hardcoded Warmup (retrieval 켜진 런 & 20000이 아닐 때)
        if ret_enable and warmup != 'N/A' and warmup != 20000:
            cell_text += f" ({warmup})"
            
        # 3. Calculated Warmup
        cell_text += f" [Calc: {calc_warmup}]"
        
        # 정렬을 위한 Numeric return 추출
        num_return = pd.to_numeric(eval_return, errors='coerce')
        if pd.isna(num_return):
            num_return = -999999
            
        data_for_df.append({
            'Game': game,
            'Config': config,
            'Seed': seed,
            'CellText': cell_text,
            'NumericReturn': num_return
        })
        
    df = pd.DataFrame(data_for_df)
    
    if df.empty:
        print("No valid data found to create excel.")
        return
        
    # 중복 런(Game, Config, Seed 동일)은 NumericReturn이 가장 높은 것으로 필터링
    df = df.sort_values(by=['Game', 'Config', 'Seed', 'NumericReturn'], ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=['Game', 'Config', 'Seed'], keep='first')
    
    # Pivot Table 생성 (행: Game & Config, 열: Seed)
    pivot_df = df.pivot_table(
        index=['Game', 'Config'],
        columns='Seed',
        values='CellText',
        aggfunc='first'
    )
    
    pivot_df = pivot_df.fillna('')
    
    # 시드(열) 정렬: 자주 쓰는 특정 시드들을 앞으로 빼고 나머지는 뒤로
    target_columns = [1, 2, 10, 710, 3710, 2000, 2010, 5090]
    existing_cols = list(pivot_df.columns)
    
    ordered_cols = [c for c in target_columns if c in existing_cols]
    extra_cols = sorted([c for c in existing_cols if c not in target_columns])
    
    # 열 사이에 빈 공간 ' ' 하나 추가 (구분선 역할, convert_csv_to_excel.py 로직 차용)
    full_columns = ordered_cols + [' '] + extra_cols 
    
    pivot_df[' '] = ''
    pivot_df = pivot_df.reindex(columns=full_columns)
    pivot_df = pivot_df.fillna('')
    
    # 인덱스 정렬: Game(알파벳순) -> Config(지정된 우선순위 순서)
    base_order = [
        'retrieval false',
        'retrieved_count (without max)',
        'retrieved_count (with max)',
        'num_valid_anchors',
        'imagine_batch_size (No retrieval deduction)'
    ]

    def get_sort_key(config_str):
        for i, base in enumerate(base_order):
            if config_str.startswith(base):
                return i
        return len(base_order)

    sorted_index = sorted(pivot_df.index, key=lambda x: (x[0], get_sort_key(x[1]), x[1]))
    pivot_df = pivot_df.reindex(sorted_index)
    
    # 엑셀 저장 및 서식 적용
    output_path = 'formula_results.xlsx'
    print(f"Generating excel file: {output_path} ...")
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        pivot_df.to_excel(writer, sheet_name='Results')
        
        worksheet = writer.sheets['Results']
        top_side = Side(border_style="medium", color="000000")
        
        start_row = worksheet.max_row - len(pivot_df) + 1
        
        for row_idx, (game, config) in enumerate(pivot_df.index):
            excel_row = row_idx + start_row
            
            # Game 이름이 바뀔 때 위쪽에 굵은 선 그리기 (가독성 목적)
            if row_idx == 0 or pivot_df.index[row_idx][0] != pivot_df.index[row_idx-1][0]:
                for col_idx in range(1, len(pivot_df.columns) + 3):
                    cell = worksheet.cell(row=excel_row, column=col_idx)
                    cell.border = Border(
                        top=top_side,
                        left=cell.border.left,
                        right=cell.border.right,
                        bottom=cell.border.bottom
                    )
                    
    # VSCode에서 바로 볼 수 있도록 Markdown 표 형식으로도 저장
    md_path = 'formula_results.md'
    print(f"Generating markdown file: {md_path} ...")
    with open(md_path, 'w', encoding='utf-8') as f:
        # 헤더 작성
        cols = ['Game', 'Config'] + list(pivot_df.columns)
        f.write('| ' + ' | '.join(str(c) for c in cols) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(cols)) + '|\n')
        
        # 데이터 작성
        for index, row in pivot_df.iterrows():
            game, config = index
            row_data = [str(game), str(config)] + [str(x) for x in row.values]
            f.write('| ' + ' | '.join(row_data) + ' |\n')
                    
    print(f"✅ Successfully generated! Check the '{output_path}' and '{md_path}' files.")
    print("💡 You can now tweak the 'calculate_warmup_steps' function and re-run this script.")

if __name__ == '__main__':
    main()
