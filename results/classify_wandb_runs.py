import sys
import os
sys.path = [p for p in sys.path if p and p != os.path.dirname(os.path.abspath(__file__))]
import wandb
import subprocess
import os
import csv
import io
import json
from collections import Counter


# W&B에서 삭제된 target: 1 (anchor 미설정) run 37개의 고정 백업.
# 출처: STORM-1/wandb_runs_classification_all.csv (2026-09-18).
# 원본의 모든 필드를 보존하며, 실행 시 백업 CSV 파일은 필요하지 않습니다.
ARCHIVED_TARGET_1_CSV = """Run Name,Run ID,State,Commit,Logic,Eval Return,Retrieval Enable,Warmup Steps,Calculated Warmup Steps,Dynamic Warmup Delay Steps,Dynamic Warmup Target Steps,Min Warmup Steps,Batch Size Reduction,Z Score Threshold,Hash Bits,Retrieval Target,Anchor Weight,Seed,Created At
Frostbite_nmi7yn4w_6000_O,nmi7yn4w,finished,486129f,num_valid_anchors,2119.5,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6000,2026-09-03T05:10:47Z
Frostbite_ykunnmjg_6010_O,ykunnmjg,finished,486129f,num_valid_anchors,261.5,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6010,2026-09-03T05:11:08Z
BattleZone_co5yu3w3_2000_O,co5yu3w3,finished,486129f,num_valid_anchors,5550,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,2000,2026-09-03T05:11:55Z
Frostbite_lby3b31o_10_O,lby3b31o,finished,486129f,num_valid_anchors,2301,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,10,2026-09-03T05:56:53Z
Frostbite_b6a6mk2q_710_O,b6a6mk2q,finished,486129f,num_valid_anchors,2609,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,710,2026-09-03T06:33:25Z
BattleZone_00jmh25e_2010_O,00jmh25e,finished,486129f,num_valid_anchors,13900,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,2010,2026-09-03T07:07:06Z
UpNDown_swshjz4c_6000_O,swshjz4c,finished,486129f,num_valid_anchors,5670,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6000,2026-09-03T07:38:12Z
UpNDown_gfu9tg3g_6010_O,gfu9tg3g,finished,486129f,num_valid_anchors,7178,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6010,2026-09-03T07:39:25Z
Frostbite_pzj25g3d_3710_O,pzj25g3d,finished,486129f,num_valid_anchors,3242.5,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,3710,2026-09-03T08:36:44Z
Qbert_ox9latfb_10_O,ox9latfb,finished,486129f,num_valid_anchors,1918.75,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,10,2026-09-03T09:19:09Z
UpNDown_x7uzwomw_6020_O,x7uzwomw,finished,486129f,num_valid_anchors,3066,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6020,2026-09-03T10:12:39Z
Qbert_ocgw5bmo_6000_O,ocgw5bmo,finished,486129f,num_valid_anchors,3782.5,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6000,2026-09-03T10:16:15Z
Qbert_2gi0no2u_1_O,2gi0no2u,finished,486129f,num_valid_anchors,2818.75,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,1,2026-09-03T11:18:08Z
Qbert_7y38q8bu_3710_O,7y38q8bu,finished,486129f,num_valid_anchors,2553.75,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,3710,2026-09-03T12:10:25Z
BattleZone_cnv22oap_6000_O,cnv22oap,finished,486129f,num_valid_anchors,10950,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6000,2026-09-03T12:44:45Z
BattleZone_yrq4azxd_6010_O,yrq4azxd,finished,486129f,num_valid_anchors,5900,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6010,2026-09-03T12:45:29Z
Krull_6foystrq_2000_O,6foystrq,finished,486129f,num_valid_anchors,4851,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,2000,2026-09-03T13:19:19Z
Alien_3810f6xi_1_O,3810f6xi,finished,486129f,num_valid_anchors,1151,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,1,2026-09-03T13:57:06Z
Amidar_zl4mdqxb_1_O,zl4mdqxb,finished,486129f,num_valid_anchors,144.6,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,1,2026-09-03T14:51:15Z
BattleZone_tmr7bbn5_6020_O,tmr7bbn5,finished,486129f,num_valid_anchors,7450,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6020,2026-09-03T15:18:56Z
Alien_jkskja9j_6000_O,jkskja9j,finished,486129f,num_valid_anchors,1303.5,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6000,2026-09-03T15:19:42Z
Krull_uk1s6s8x_2010_O,uk1s6s8x,finished,486129f,num_valid_anchors,6044,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,2010,2026-09-03T16:00:44Z
Assault_e6itftuv_1_O,e6itftuv,finished,486129f,num_valid_anchors,740.75,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,1,2026-09-03T16:37:01Z
Asterix_h7aofkuv_10_O,h7aofkuv,finished,486129f,num_valid_anchors,467.5,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,10,2026-09-03T17:32:10Z
Alien_7lc9bxth_6010_O,7lc9bxth,finished,486129f,num_valid_anchors,1032,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6010,2026-09-03T17:52:24Z
Amidar_jbm316dm_6000_O,jbm316dm,finished,486129f,num_valid_anchors,211.7,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6000,2026-09-03T17:52:56Z
Asterix_toxgeo04_3710_O,toxgeo04,finished,486129f,num_valid_anchors,677.5,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,3710,2026-09-03T19:16:44Z
Boxing_zsaukhwx_10_O,zsaukhwx,finished,486129f,num_valid_anchors,74.85,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,10,2026-09-03T20:13:21Z
Amidar_rxipad36_6010_O,rxipad36,finished,486129f,num_valid_anchors,124.75,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6010,2026-09-03T20:22:34Z
Amidar_ginz9ksm_6020_O,ginz9ksm,finished,486129f,num_valid_anchors,159.3,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6020,2026-09-03T20:25:20Z
Alien_ts6fn9ut_2000_O,ts6fn9ut,finished,486129f,num_valid_anchors,641,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,2000,2026-09-03T21:18:33Z
Breakout_ix0b33ni_10_O,ix0b33ni,finished,486129f,num_valid_anchors,12.1,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,10,2026-09-03T21:55:23Z
Breakout_hkojwxtr_3710_O,hkojwxtr,finished,486129f,num_valid_anchors,15.6,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,3710,2026-09-03T22:55:26Z
Assault_piki59lv_6000_O,piki59lv,finished,486129f,num_valid_anchors,1033.7,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6000,2026-09-03T22:55:34Z
Assault_holuktxi_6010_O,holuktxi,finished,486129f,num_valid_anchors,867.85,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,6010,2026-09-03T22:58:32Z
Assault_d0c7w16z_2000_O,d0c7w16z,finished,486129f,num_valid_anchors,693.3,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,2000,2026-09-03T23:48:23Z
ChopperCommand_as286v3o_710_O,as286v3o,finished,486129f,num_valid_anchors,1360,True,50000,50000,3000,20000,5000,retrieved,3.5,11,1,N/A,710,2026-09-04T00:34:10Z
"""


def merge_archived_target_1_runs(results):
    """W&B 결과에 없는 백업 Run ID만 추가하며 동일 ID는 W&B 값을 우선합니다."""
    live_ids = {row["Run ID"] for row in results}
    archived_rows = csv.DictReader(io.StringIO(ARCHIVED_TARGET_1_CSV))
    return list(results) + [row for row in archived_rows if row["Run ID"] not in live_ids]


def get_logic_for_commit(commit_hash):
    """
    주어진 커밋 해시에서 train.py 파일을 읽어와 
    random_batch_size 설정 로직을 파악합니다.
    """
    if not commit_hash:
        return "Unknown (No commit hash)"
    
    try:
        # git show <commit>:train.py 명령어로 해당 커밋 시점의 코드 내용 추출
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:train.py"],
            capture_output=True, text=True, check=True
        )
        code = result.stdout
        
        # 코드 내용에 따른 분류
        # 최신 코드의 경우 num_valid_anchors와 retrieved_count 조건이 모두 포함될 수 있으므로, 두 개가 다 있을 땐 num_valid_anchors로 분류
        if "random_batch_size = max(0, imagine_batch_size - retrieved_count)" in code and "random_batch_size = max(0, imagine_batch_size - num_valid_anchors)" in code:
            return "num_valid_anchors"
        elif "random_batch_size = imagine_batch_size - num_valid_anchors" in code or "random_batch_size = max(0, imagine_batch_size - num_valid_anchors)" in code:
            return "num_valid_anchors"
        elif "random_batch_size = max(0, imagine_batch_size - retrieved_count)" in code:
            return "retrieved_count (with max)"
        elif "random_batch_size = imagine_batch_size - retrieved_count" in code:
            return "retrieved_count (without max)"
        elif "random_batch_size = imagine_batch_size" in code:
            return "imagine_batch_size (No retrieval deduction)"
        else:
            return "Unknown (Logic not found in code)"
            
    except subprocess.CalledProcessError:
        # 파일이 존재하지 않거나 커밋을 찾을 수 없는 경우
        return "Unknown (Git error or train.py missing)"

def main():
    # 1. 자동 로그인 처리
    api_key_path = os.path.join(os.path.dirname(__file__), '.wandb_api_key')
    if os.path.exists(api_key_path):
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
            wandb.login(key=api_key)
            print("✅ .wandb_api_key를 사용하여 성공적으로 로그인했습니다.")
    else:
        print("⚠️ .wandb_api_key 파일을 찾을 수 없습니다. 기존 설정된 자격 증명을 시도합니다.")

    # 2. WandB API 객체 생성 및 엔티티 자동 추출
    api = wandb.Api()
    try:
        # 명시적으로 choemj-kaist 엔티티 사용
        entity = "choemj-kaist"
        wandb_path = f"{entity}/STORM"
        print(f"'{wandb_path}' 프로젝트의 run들을 분석합니다...")
        runs = api.runs(wandb_path)
    except Exception as e:
        print(f"WandB API 호출 실패: {e}")
        return

    results = []
    
    print(f"총 {len(runs)}개의 run을 확인했습니다. 분류를 시작합니다...\n")
    
    def get_config_val(config, key_path):
        if key_path in config:
            return config[key_path]
        keys = key_path.split('.')
        val = config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return None
        return val

    for run in runs:
        if run.state == "killed":
            continue

        ret_enable = get_config_val(run.config, 'JointTrainAgent.Retrieval.enable')
        is_both = (
            str(ret_enable).strip().lower() == 'both'
            or str(run.name).lower().endswith('_both')
        )
        is_experiment_list = isinstance(ret_enable, list)
        if (is_both or is_experiment_list) and run.state != "running":
            continue
        if is_both:
            ret_enable = 'Both'
        elif not is_experiment_list:
            ret_enable = str(ret_enable).strip().lower() in ('true', '1', 't')
        save_warmup = get_config_val(run.config, 'JointTrainAgent.Retrieval.save_warmup')
            
        # WandB는 기본적으로 github 연동이나 git 추적 시 commit 정보를 남깁니다.
        commit_hash = run.commit
        
        # run.commit이 없는 경우 config나 summary 등 다른 곳에 수동 기록했는지 확인
        if not commit_hash and 'commit' in run.config:
            commit_hash = run.config['commit']
            
        logic_type = get_logic_for_commit(commit_hash)
        
        eval_return = run.summary.get('eval/episode_avg_return', 'N/A')
        
        warmup_steps = 'N/A'
        calculated_warmup_steps = 'N/A'
        if ret_enable:
            w_steps = get_config_val(run.config, 'JointTrainAgent.Retrieval.warmup_steps')
            warmup_steps = w_steps if w_steps is not None else 'N/A'
            calculated_warmup_steps = warmup_steps
            
            # warmup_steps가 -1인 경우, Retrieval/retrieved_contexts 그래프의 첫 스텝에서 가져옴
            if str(warmup_steps) == '-1':
                try:
                    # 백엔드 샘플링을 방지하기 위해 samples 값을 크게 설정하여 전체를 가져옵니다.
                    hist_df = run.history(keys=["Retrieval/retrieved_contexts"], samples=1000000)
                    if not hist_df.empty and "_step" in hist_df.columns:
                        first_step = int(hist_df["_step"].min())
                        calculated_warmup_steps = first_step + 1024
                    else:
                        print(f"[{run.name}] 히스토리에 '_step' 정보가 없습니다.")
                except Exception as e:
                    print(f"[{run.name}] 히스토리에서 warmup steps를 가져오지 못했습니다: {e}")
            
        # 1. 런 이름에서 시드 추출 우선 시도 (형식: {env}_{id}_{seed}_{O/X/Both})
        seed = None
        parts = str(run.name).split('_')
        if len(parts) >= 4 and parts[-1].upper() in ['O', 'X', 'BOTH']:
            try:
                seed = int(parts[-2])
            except ValueError:
                pass
        elif len(parts) >= 3 and parts[-1].isdigit():
            # Both 공통 단계의 Logger는 O/X 접미사 없이 이름을 기록합니다.
            seed = int(parts[-1])
                
        # 2. 런 이름에서 유추 실패 시 Config에서 읽어오기 ('Seed' 또는 'seed' 확인)
        if seed is None:
            seed = get_config_val(run.config, 'Seed')
            if seed is None:
                seed = get_config_val(run.config, 'seed')
                
        seed = seed if seed is not None else 'N/A'
        
        dynamic_warmup_delay_steps = get_config_val(run.config, 'JointTrainAgent.Retrieval.dynamic_warmup_delay_steps')
        dynamic_warmup_delay_steps = dynamic_warmup_delay_steps if dynamic_warmup_delay_steps is not None else 'N/A'
        
        dynamic_warmup_target_steps = get_config_val(run.config, 'JointTrainAgent.Retrieval.dynamic_warmup_target_steps')
        dynamic_warmup_target_steps = dynamic_warmup_target_steps if dynamic_warmup_target_steps is not None else 'N/A'
        
        min_warmup_steps = get_config_val(run.config, 'JointTrainAgent.Retrieval.min_warmup_steps')
        min_warmup_steps = min_warmup_steps if min_warmup_steps is not None else 'N/A'
        
        batch_size_reduction = get_config_val(run.config, 'JointTrainAgent.Retrieval.batch_size_reduction')
        batch_size_reduction = batch_size_reduction if batch_size_reduction is not None else 'N/A'
        
        z_score_threshold = get_config_val(run.config, 'JointTrainAgent.Retrieval.z_score_threshold')
        z_score_threshold = z_score_threshold if z_score_threshold is not None else 'N/A'

        value_signal = get_config_val(run.config, 'JointTrainAgent.Retrieval.value_signal')
        score_combination = get_config_val(run.config, 'JointTrainAgent.Retrieval.score_combination')
        additive_z_score_threshold = get_config_val(run.config, 'JointTrainAgent.Retrieval.additive_z_score_threshold')
        
        hash_bits = get_config_val(run.config, 'JointTrainAgent.Retrieval.hash_bits')
        hash_bits = hash_bits if hash_bits is not None else 'N/A'
        
        retrieval_target = get_config_val(run.config, 'JointTrainAgent.Retrieval.target')
        retrieval_target = retrieval_target if retrieval_target is not None else 'N/A'
        
        anchor_weight = get_config_val(run.config, 'JointTrainAgent.Retrieval.anchor_weight')
        anchor_weight = anchor_weight if anchor_weight is not None else 'N/A'
        
        results.append({
            "Run Name": run.name,
            "Run ID": run.id,
            "State": run.state,
            "Commit": commit_hash[:7] if commit_hash else "None",
            "Logic": logic_type,
            "Eval Return": eval_return,
            "Retrieval Enable": json.dumps(ret_enable) if isinstance(ret_enable, list) else ret_enable,
            "Save Warmup": save_warmup if save_warmup is not None else 'N/A',
            "Warmup Steps": warmup_steps,
            "Calculated Warmup Steps": calculated_warmup_steps,
            "Dynamic Warmup Delay Steps": dynamic_warmup_delay_steps,
            "Dynamic Warmup Target Steps": dynamic_warmup_target_steps,
            "Min Warmup Steps": min_warmup_steps,
            "Batch Size Reduction": batch_size_reduction,
            "Z Score Threshold": z_score_threshold,
            "Value Signal": value_signal if value_signal is not None else 'N/A',
            "Score Combination": score_combination if score_combination is not None else 'N/A',
            "Additive Z Score Threshold": additive_z_score_threshold if additive_z_score_threshold is not None else 'N/A',
            "Hash Bits": hash_bits,
            "Retrieval Target": retrieval_target,
            "Anchor Weight": anchor_weight,
            "Seed": seed,
            "Created At": run.created_at
        })
        print(f"Run: {run.name:20} | Commit: {str(commit_hash)[:7]:7} | Return: {str(eval_return)[:8]:8} | Logic: {logic_type} | Ret: {ret_enable} | Orig Warmup: {warmup_steps} | Calc Warmup: {calculated_warmup_steps} | Seed: {seed} | Created: {run.created_at}")

    live_count = len(results)
    results = merge_archived_target_1_runs(results)
    print(f"\n고정 백업에서 target: 1 (anchor 미설정) run {len(results) - live_count}개를 추가했습니다.")
    logic_counts = Counter(row["Logic"] for row in results)

    # 요약 및 저장
    print("\n" + "="*50)
    print("분류 요약:")
    for logic, count in logic_counts.most_common():
        print(f"  {logic}: {count} runs")
    print("="*50)
    
    output_csv = "wandb_runs_classification.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Run Name", "Run ID", "State", "Commit", "Logic", "Eval Return", "Retrieval Enable", "Save Warmup", "Warmup Steps", "Calculated Warmup Steps", "Dynamic Warmup Delay Steps", "Dynamic Warmup Target Steps", "Min Warmup Steps", "Batch Size Reduction", "Z Score Threshold", "Value Signal", "Score Combination", "Additive Z Score Threshold", "Hash Bits", "Retrieval Target", "Anchor Weight", "Seed", "Created At"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"\n상세 결과가 '{output_csv}' 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
