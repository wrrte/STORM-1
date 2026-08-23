import numpy as np
import cv2
import os

def _is_submarine(lane_idx, ram):
    """
    OCAtari 기준: 해당 라인의 적이 Submarine인지 확인.
    Submarine이면 해당 라인의 물체는 미사일이고, 아니면(Shark이면) 잠수부임.
    lane_idx는 0~3.
    """
    return 3 < (ram[89 + lane_idx] % 8) < 7

def main():
    demo_path = '/media/storage_data/ai2lab/choemj/Drama_modified/demonstrations/Seaquest_20260729_225940.npz'
    print(f"Loading {demo_path}...")
    data = np.load(demo_path)
    obs = data['obs']  # (N, 64, 64, 3)
    ram = data['ram']  # (N, 128)
    
    out_dir = "diver_verification_fixed"
    os.makedirs(out_dir, exist_ok=True)
    
    # 잠수부 획득 프레임 찾기 (픽업 직전 프레임들)
    diver_count = ram[:, 62]
    pickup_frames = np.where(np.diff(diver_count) == 1)[0]
    
    # 10프레임 간격으로 20장 추출 (다양한 상황 확인용)
    sample_frames = list(range(0, min(len(obs), 200), 10))
    
    for idx, f_idx in enumerate(sample_frames):
        img = obs[f_idx].copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        current_ram = ram[f_idx]

        # 오직 잠수부(Diver)만 찾아서 빨간색 동그라미 치기
        # Seaquest에는 4개의 레인(Lane)이 있음
        for i in range(4):
            lane_idx = 3 - i
            diver_x = current_ram[74 - i]
            
            # X 좌표가 0 초과 160 미만인 경우에만 화면에 존재
            if 0 < diver_x < 160:
                # 해당 레인의 적이 잠수함이 아니어야 잠수부임 (잠수함이면 적의 미사일)
                if not _is_submarine(lane_idx, current_ram):
                    # Y 좌표는 레인별로 고정되어 있음 (141, 117, 93, 69)
                    diver_y = 141 - lane_idx * 24
                    
                    px = int(diver_x / 160.0 * 256)
                    py = int(diver_y / 210.0 * 256)
                    
                    # 빨간색 동그라미로 잠수부 표시
                    cv2.circle(img, (px, py), 12, (0, 0, 255), 2)
                    cv2.putText(img, f"Diver", (px-15, py-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        out_path = os.path.join(out_dir, f"frame_{f_idx:04d}.png")
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")
        
    print(f"\nOCAtari 기반 시각화 완료! '{out_dir}' 폴더를 확인해주세요.")

if __name__ == "__main__":
    main()
