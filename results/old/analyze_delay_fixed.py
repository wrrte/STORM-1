import pandas as pd
import numpy as np
import os

# 이미지에서 추출한 Game별 Random, Human Score 데이터
HNS_DATA = {
    'Alien': {'random': 228, 'human': 7128},
    'Amidar': {'random': 6, 'human': 1720},
    'Assault': {'random': 222, 'human': 742},
    'Asterix': {'random': 210, 'human': 8503},
    'BankHeist': {'random': 14, 'human': 753},
    'BattleZone': {'random': 2360, 'human': 37188},
    'Boxing': {'random': 0, 'human': 12},
    'Breakout': {'random': 2, 'human': 30},
    'ChopperCommand': {'random': 811, 'human': 7388},
    'CrazyClimber': {'random': 10780, 'human': 35829},
    'DemonAttack': {'random': 152, 'human': 1971},
    'Freeway': {'random': 0, 'human': 30},
    'Frostbite': {'random': 65, 'human': 4335},
    'Gopher': {'random': 258, 'human': 2413},
    'Hero': {'random': 1027, 'human': 30826},
    'Jamesbond': {'random': 29, 'human': 303},
    'Kangaroo': {'random': 52, 'human': 3035},
    'Krull': {'random': 1598, 'human': 2666},
    'KungFuMaster': {'random': 256, 'human': 22736},
    'MsPacman': {'random': 307, 'human': 6952},
    'Pong': {'random': -21, 'human': 15},
    'Qbert': {'random': 164, 'human': 13455},
    'Seaquest': {'random': 68, 'human': 42055},
    'UpNDown': {'random': 533, 'human': 11693},
}

def main():
    csv_path = '../wandb_runs_classification.csv'
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    data = []
    for idx, row in df.iterrows():
        run_name = str(row['Run Name'])
        parts = run_name.split('_')
        game = parts[0]
        
        seed = str(row['Seed'])
        if seed.endswith('.0'): seed = seed[:-2]
        if seed in ['N/A', 'nan', 'None']:
            if len(parts) >= 3 and parts[2].isdigit():
                seed = parts[2]
            elif len(parts) == 4 and parts[3].isdigit():
                seed = parts[3]
        if not str(seed).isdigit(): continue
        seed = int(seed)
        
        ret_enable = str(row['Retrieval Enable']).strip().lower() in ['true', '1', 't']
        
        # convert_csv_to_excel.py 와 동일한 로직 사용
        created_at_utc = pd.to_datetime(row.get('Created At', '2000-01-01T00:00:00Z'), utc=True)
        is_new_code = created_at_utc >= pd.to_datetime("2026-08-20T10:00:00Z", utc=True)
        
        if not ret_enable:
            config = 'Retrieval 미사용'
        elif is_new_code:
            delay = row.get('Dynamic Warmup Delay Steps', 'N/A')
            if delay != 'N/A' and not pd.isna(delay):
                config = f'최신 코드 (8/20 19:00 이후) - Delay: {int(float(delay))}'
            else:
                config = '최신 코드 (8/20 19:00 이후)'
        else:
            logic = str(row['Logic']).strip()
            if 'retrieved_count' in logic:
                config = '기존 retrieved_count'
            else:
                continue # num_valid_anchors 등은 무시 (convert_csv_to_excel.py 와 동일)
                
        eval_return = row['Eval Return']
        try:
            eval_return = float(eval_return)
            if pd.isna(eval_return): continue
        except ValueError:
            continue
            
        # Human Normalized Score (HNS) 계산
        hns = np.nan
        if game in HNS_DATA:
            r = HNS_DATA[game]['random']
            h = HNS_DATA[game]['human']
            if h - r != 0:
                hns = (eval_return - r) / (h - r) * 100.0
            
        data.append({
            'Game': game,
            'Config': config,
            'Seed': seed,
            'HNS': hns,
            'Created At': created_at_utc
        })

    parsed_df = pd.DataFrame(data)
    if parsed_df.empty:
        print("No valid data to analyze.")
        return

    # NaN HNS 제외
    parsed_df = parsed_df.dropna(subset=['HNS'])

    # 최신 런 우선, Game/Config/Seed 중복 시 첫번째(최신) 유지
    parsed_df = parsed_df.sort_values(by='Created At', ascending=False)
    parsed_df = parsed_df.drop_duplicates(subset=['Game', 'Config', 'Seed'], keep='first')

    # Seed 묶어서 평균
    mean_df = parsed_df.groupby(['Game', 'Config'])['HNS'].mean().reset_index()

    # 피벗 테이블
    pivot_df = mean_df.pivot(index='Game', columns='Config', values='HNS')
    
    # 열 정렬
    base_order = ['Retrieval 미사용', '기존 retrieved_count']
    delay_cols = sorted([c for c in pivot_df.columns if 'Delay:' in c], key=lambda x: int(x.split('Delay: ')[1]))
    ordered_cols = [c for c in base_order if c in pivot_df.columns] + delay_cols
    pivot_df = pivot_df[ordered_cols]
    
    print("\n[ 평균 Human Normalized Score (Game vs Config) ]")
    print("-" * 100)
    # 소수점 2자리까지만 출력
    pd.options.display.float_format = '{:.2f}%'.format
    print(pivot_df.to_string(na_rep='-'))
    print("-" * 100)

    # -------------------------------------------------------------------------
    # 정확한 1:1 시드(Seed) 비교를 위한 추가 피벗 테이블
    # (Game, Seed) 쌍을 기준으로 두 Config가 모두 플레이한 공통 시드만 추출하기 위함
    # -------------------------------------------------------------------------
    seed_pivot_df = parsed_df.pivot(index=['Game', 'Seed'], columns='Config', values='HNS')

    # Oracle(기존 retrieved_count) vs No Retrieval 비교
    print("\n[ 오라클(기존 retrieved_count) vs Retrieval 미사용 HNS 요약 ]")
    base_oracle = '기존 retrieved_count'
    base_no_ret = 'Retrieval 미사용'
    
    if base_oracle in seed_pivot_df.columns and base_no_ret in seed_pivot_df.columns:
        # 두 Config가 모두 플레이한 (Game, Seed) 쌍만 정확히 추려냄
        paired_base = seed_pivot_df[[base_oracle, base_no_ret]].dropna()
        
        if len(paired_base) > 0:
            # 1:1 매칭된 시드별로 승/무/패 판정
            wins_b = (paired_base[base_oracle] > paired_base[base_no_ret]).sum()
            losses_b = (paired_base[base_oracle] < paired_base[base_no_ret]).sum()
            ties_b = (paired_base[base_oracle] == paired_base[base_no_ret]).sum()
            
            # 정확히 1:1 매칭된 시드들만의 평균 계산
            mean_oracle = paired_base[base_oracle].mean()
            mean_no_ret = paired_base[base_no_ret].mean()
            
            print(f"▶ {base_oracle}")
            print(f"  vs {base_no_ret:25s} | Win: {wins_b:2d}, Loss: {losses_b:2d}, Tie: {ties_b:2d} | Avg HNS: {mean_oracle:.2f}% vs {mean_no_ret:.2f}% (공통 시드 1:1 비교)")
            
            # 가장 불리한 점수 차이가 나온 게임과 시드 찾기
            diff = paired_base[base_oracle] - paired_base[base_no_ret]
            worst_idx = diff.idxmin()
            worst_game, worst_seed = worst_idx
            worst_oracle = paired_base.loc[worst_idx, base_oracle]
            worst_no_ret = paired_base.loc[worst_idx, base_no_ret]
            print(f"    [최악의 하락] Game: {worst_game}, Seed: {worst_seed} | 오라클: {worst_oracle:.2f}% vs 미사용: {worst_no_ret:.2f}% (차이: {diff[worst_idx]:.2f}%p)")
        else:
            print("  비교 가능한 공통 시드가 없음")
    else:
        print("  해당 베이스라인 데이터가 없어 비교할 수 없습니다.")

    # Delay별로 Baseline(Retrieval 미사용, 기존 retrieved_count)과 성능 비교
    print("\n[ 고정 Delay 설정별 HNS 요약 ]")
    
    for delay in delay_cols:
        print(f"\n▶ {delay}")
        for base in base_order:
            if base not in seed_pivot_df.columns or delay not in seed_pivot_df.columns:
                continue
                
            # 정확히 1:1 매칭되는 (Game, Seed) 쌍만 추출
            paired = seed_pivot_df[[delay, base]].dropna()
            
            if len(paired) == 0:
                print(f"  vs {base:25s} | 비교 가능한 공통 시드 없음")
                continue
                
            wins = (paired[delay] > paired[base]).sum()
            losses = (paired[delay] < paired[base]).sum()
            ties = (paired[delay] == paired[base]).sum()
            
            mean_delay = paired[delay].mean()
            mean_base = paired[base].mean()
            
            print(f"  vs {base:25s} | Win: {wins:2d}, Loss: {losses:2d}, Tie: {ties:2d} | Avg HNS: {mean_delay:.2f}% vs {mean_base:.2f}% (공통 시드 1:1 비교)")

if __name__ == '__main__':
    main()
