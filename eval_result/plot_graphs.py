import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Overleaf 논문용으로 깔끔하고 전문적인 스타일 설정
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# 색상 팔레트 설정 (Baseline은 옅은 회색/파랑, FLASH는 강조되는 빨강/주황)
COLOR_BASELINE = '#8c96c6' 
COLOR_FLASH = '#e34a33'

def plot_distribution():
    print("Generating Final Distribution Plot (Box + Swarm)...")
    try:
        df_o = pd.read_csv('flash_o_100000_each.csv')
        df_x = pd.read_csv('flash_x_100000_each.csv')
    except Exception as e:
        print(f"Error loading distribution data: {e}")
        return

    # 마지막 스텝(가장 아래 행) 데이터 추출
    last_row_o = df_o.iloc[-1]
    last_row_x = df_x.iloc[-1]
    
    # __MIN, __MAX가 포함되지 않은 순수 return 열만 필터링
    cols_o = [c for c in df_o.columns if 'return' in c and '__' not in c]
    cols_x = [c for c in df_x.columns if 'return' in c and '__' not in c]
    
    vals_o = last_row_o[cols_o].values.astype(float)
    vals_x = last_row_x[cols_x].values.astype(float)
    
    # Seaborn 플로팅을 위한 데이터프레임 구성
    data = []
    for v in vals_x:
        data.append({'Method': 'Baseline', 'Score': v})
    for v in vals_o:
        data.append({'Method': 'FLASH', 'Score': v})
    
    df_plot = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {'Baseline': COLOR_BASELINE, 'FLASH': COLOR_FLASH}
    
    # 1. Boxplot (투명하게 배경으로 깔기)
    sns.boxplot(data=df_plot, x='Method', y='Score', 
                palette=palette, width=0.4, 
                boxprops=dict(alpha=0.3, edgecolor='black', linewidth=1.5),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5),
                showfliers=False, ax=ax)
                
    # 2. Swarmplot (데이터 포인트 20개 찍기)
    sns.swarmplot(data=df_plot, x='Method', y='Score', 
                  palette=palette, size=8, edgecolor='black', linewidth=1, ax=ax)
                  
    ax.set_ylabel('Final Evaluation Score', fontweight='bold')
    ax.set_xlabel('')
    
    # 테두리 정리
    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig('fig_distribution.pdf', format='pdf', dpi=300, bbox_inches='tight')
    print("Saved -> fig_distribution.pdf")

def plot_learning_curves():
    print("Generating Learning Curves...")
    
    # 예시: 성공한 시드의 Eval Score 그래프 (제공된 csv 기반)
    try:
        eval_o = pd.read_csv('Frostbite-life_done-wm_2L512D8H-100k-seed10_O_20260828_120823.csv')
        eval_x = pd.read_csv('Frostbite-life_done-wm_2L512D8H-100k-seed710_X_20260828_114604.csv')
        
        # 정렬
        eval_o = eval_o.sort_values(eval_o.columns[0]) # step column
        eval_x = eval_x.sort_values(eval_x.columns[0])
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(eval_x.iloc[:, 0], eval_x.iloc[:, 1], color=COLOR_BASELINE, linewidth=2.5, label='Baseline')
        ax.plot(eval_o.iloc[:, 0], eval_o.iloc[:, 1], color=COLOR_FLASH, linewidth=2.5, label='FLASH')
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Evaluation Score')
        ax.legend()
        sns.despine()
        plt.tight_layout()
        plt.savefig('fig_eval_success.pdf', format='pdf', dpi=300, bbox_inches='tight')
        print("Saved -> fig_eval_success.pdf")
        
    except Exception as e:
        print(f"Error loading eval data: {e}")

    # 예시: 성공한 시드의 Episode Score 그래프
    try:
        ep_o = pd.read_csv('flash_o_episode.csv')
        ep_x = pd.read_csv('flash_x_episode.csv')
        
        col_o = [c for c in ep_o.columns if 'reward' in c and '__' not in c][0]
        col_x = [c for c in ep_x.columns if 'reward' in c and '__' not in c][0]
        
        fig, ax = plt.subplots(figsize=(5, 4))
        # 노이즈가 많으므로 alpha 값을 주어 덜 복잡해 보이게 함
        ax.plot(ep_x['Step'], ep_x[col_x], color=COLOR_BASELINE, linewidth=1.5, alpha=0.7, label='Baseline')
        ax.plot(ep_o['Step'], ep_o[col_o], color=COLOR_FLASH, linewidth=1.5, alpha=0.7, label='FLASH')
        
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Episode Score')
        ax.legend()
        sns.despine()
        plt.tight_layout()
        plt.savefig('fig_episode_success.pdf', format='pdf', dpi=300, bbox_inches='tight')
        print("Saved -> fig_episode_success.pdf")
    except Exception as e:
        print(f"Error loading episode data: {e}")

if __name__ == '__main__':
    plot_distribution()
    plot_learning_curves()
    print("Done! You can upload the .pdf files to Overleaf.")
