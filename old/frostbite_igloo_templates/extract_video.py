import numpy as np
import imageio
import os

# 파일 경로 설정
npz_path = 'probing_data/Seaquest-life_done-wm_2L512D8H-100k-seed1_X_probing.npz'
out_path = 'probing_data/seaquest_sample_crisp.mp4'

print(f"Loading {npz_path}...")
data = np.load(npz_path)
obs = data['obs']

print(f'obs shape: {obs.shape}, dtype: {obs.dtype}')

# 60초 분량 추출 (30fps 기준 1800 프레임)
num_frames = min(1800, obs.shape[0])
frames = obs[:num_frames]

# 이미지 저장을 위해 0~255 범위의 uint8 타입으로 변환
if frames.dtype == np.float32 and frames.max() <= 1.0:
    frames = (frames * 255).astype(np.uint8)
elif frames.dtype == np.float32 and frames.max() > 1.0:
    frames = frames.astype(np.uint8)

# ⭐️ 해상도 8배 뻥튀기 (Nearest-Neighbor 보간법과 동일한 효과)
# (1800, 64, 64, 3) -> (1800, 512, 512, 3)
scale_factor = 8
frames_scaled = np.repeat(np.repeat(frames, scale_factor, axis=1), scale_factor, axis=2)

print(f"Saving video to {out_path}...")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
# 압축 과정에서 흐려지는 것을 막기 위해 높은 quality 옵션(1~10) 사용
imageio.mimsave(out_path, frames_scaled, fps=30, quality=10)
print("Done!")
