import numpy as np
import cv2
import os

# 64x64 입력 이미지 기준 예상 Bounding Box
X_MIN = 18
Y_MIN = 53
X_MAX = 46
Y_MAX = 57

def main():
    data_path = "probing_data/Seaquest_20260822_133245.npz"
    data = np.load(data_path)
    obs = data["obs"]  # (N, 64, 64, 3)
    ram = data["ram"]
    
    diver_counts = ram[:, 62]
    
    out_dir = "bottom_diver_bboxes"
    os.makedirs(out_dir, exist_ok=True)
    
    found = set()
    # 개수가 바뀐 직후에는 화면 렌더링 딜레이나 깜빡임이 있을 수 있으므로, 
    # 개수가 변하고 20프레임이 지난 안정적인 상태의 이미지를 캡처합니다.
    for i in range(20, len(obs)):
        c = diver_counts[i]
        if c not in found and c <= 6 and diver_counts[i-20] == c:
            img = obs[i].copy()
            # BGR 변환
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 사각형 그리기 (빨간색)
            cv2.rectangle(img, (X_MIN, Y_MIN), (X_MAX, Y_MAX), (0, 0, 255), 1)
            
            # 확대해서 보기 편하게 256x256으로 리사이즈
            img_large = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
            
            out_path = os.path.join(out_dir, f"diver_count_{c}.png")
            cv2.imwrite(out_path, img_large)
            print(f"저장 완료: {out_path} (잠수부 {c}명)")
            found.add(c)
            
        if len(found) == 7:
            break

    print(f"\n모든 이미지 저장이 완료되었습니다. {out_dir}/ 디렉토리를 확인해주세요.")
    print(f"현재 64x64 기준 Bounding Box: X({X_MIN}~{X_MAX}), Y({Y_MIN}~{Y_MAX})")

if __name__ == "__main__":
    main()
