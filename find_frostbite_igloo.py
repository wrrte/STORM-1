import numpy as np
import cv2
import os

def find_igloo_ram_index(ram):
    """
    이글루 블록 수는 0부터 시작해서 점진적으로 증가하는 특징이 있습니다.
    """
    candidates = []
    for i in range(128):
        unique_vals = np.unique(ram[:, i])
        max_val = np.max(ram[:, i])
        
        # 유니크한 값이 10개에서 40개 사이인 경우 (17단계 이상일 수도 있으므로 여유 있게)
        if 10 <= len(unique_vals) <= 40:
            diffs = np.diff(ram[:, i])
            num_increases = np.sum(diffs > 0)
            num_decreases = np.sum(diffs < 0)
            
            # 주로 증가하는 패턴인 경우 후보로 등록
            if num_increases > 5 and num_decreases < num_increases * 0.5:
                candidates.append((i, len(unique_vals), max_val))
                
    # 증가하는 횟수가 많은 순으로 정렬 (가장 이글루다운 것)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates], candidates

def main():
    data_path = "probing_data/Frostbite_20260822_201417.npz"
    print(f"데이터 로드 중: {data_path}")
    data = np.load(data_path)
    obs = data['obs']  # 예상: (N, 64, 64, 3)
    ram = data['ram']  # 예상: (N, 128)
    
    print(f"obs 형태: {obs.shape}")
    print(f"ram 형태: {ram.shape}")
    
    candidate_indices, detailed_candidates = find_igloo_ram_index(ram)
    print(f"이글루 RAM 인덱스 유력 후보들 (Index, Unique개수, Max값):")
    for c in detailed_candidates:
        print(f"  - RAM[{c[0]}]: {c[1]} stages, max={c[2]}")
    
    if not candidate_indices:
        print("RAM 인덱스를 찾지 못했습니다.")
        return
        
    # RAM 77이 Frostbite에서 이글루 블록(총 16~17단계)을 의미할 확률이 매우 높습니다.
    best_idx = 77
    print(f"\n최우선 채택된 RAM 인덱스: {best_idx}")
    
    igloo_counts = ram[:, best_idx]
    unique_counts = sorted(np.unique(igloo_counts))
    print(f"발견된 이글루 단계: 총 {len(unique_counts)}단계 {unique_counts}")
    
    out_dir = "frostbite_igloo_bbox"
    os.makedirs(out_dir, exist_ok=True)
    
    # 이전 실행의 잔여 파일들 삭제
    import glob
    for f in glob.glob(os.path.join(out_dir, "*.png")):
        os.remove(f)
    
    # BBox 픽셀 미세 조정 (위쪽 1픽셀 늘림)
    X_MIN, X_MAX = 42, 58
    Y_MIN, Y_MAX = 10, 17
    
    found_counts = set()
    # 0부터 최대 단계까지 하나씩 찾아서 이미지 저장
    for i in range(len(obs)):
        c = igloo_counts[i]
        if c not in found_counts:
            # 화면 안정화를 위해 값이 바뀌고 5프레임 뒤 캡처
            stable_idx = min(i + 5, len(obs) - 1)
            
            # 혹시 그 5프레임 사이에 다시 바뀌진 않았는지 확인
            if igloo_counts[stable_idx] == c:
                img = obs[stable_idx].copy()
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # BBox 그리기 (빨간색)
                cv2.rectangle(img, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 0, 255), 1)
                
                # 4배 확대 (256x256)
                img_large = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
                
                # RAM 255는 0단계(아무것도 없음)를 의미하며, RAM 0~15는 1~16단계를 의미함
                if c == 255:
                    real_stage = 0
                else:
                    real_stage = c + 1
                    
                out_path = os.path.join(out_dir, f"igloo_stage_{real_stage:02d}.png")
                cv2.imwrite(out_path, img_large)
                print(f"저장 완료: {out_path} (RAM 값: {c} -> 실제 단계: {real_stage})")
                
                found_counts.add(c)
                
    print(f"\n모든 단계 이미지 저장이 완료되었습니다. {out_dir}/ 디렉토리를 확인해주세요.")
    print(f"현재 64x64 기준 Bounding Box: X({X_MIN}~{X_MAX}), Y({Y_MIN}~{Y_MAX})")

if __name__ == "__main__":
    main()
