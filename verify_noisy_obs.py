import numpy as np
import cv2
import os

def inject_background_noise(obs_array):
    noise = np.random.randint(0, 256, size=obs_array.shape, dtype=np.uint8)
    mask = np.zeros_like(obs_array)
    # 하단 잠수부 BBox 영역(X: 18~46, Y: 53~57)만 원본 유지.
    mask[:, 53:57, 18:46, :] = 1
    
    noisy_obs = obs_array * mask + noise * (1 - mask)
    return noisy_obs

def main():
    data_path = "probing_data/Seaquest_20260822_133245.npz"
    print("데이터 로드 중...")
    data = np.load(data_path)
    obs = data["obs"]  # (N, 64, 64, 3)
    
    print("노이즈 주입 중...")
    noisy_obs = inject_background_noise(obs)
    
    out_dir = "noisy_images"
    os.makedirs(out_dir, exist_ok=True)
    
    # 몇 개의 샘플 프레임 저장
    sample_indices = [100, 1000, 5000, 10000, 20000]
    for idx in sample_indices:
        img = noisy_obs[idx].copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # 보기 편하게 4배 확대
        img_large = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
        out_path = os.path.join(out_dir, f"noisy_frame_{idx}.png")
        cv2.imwrite(out_path, img_large)
        print(f"저장 완료: {out_path}")
        
    print(f"\n모든 노이즈 이미지 저장이 완료되었습니다. {out_dir}/ 디렉토리를 확인해주세요.")

if __name__ == "__main__":
    main()
