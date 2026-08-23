import numpy as np
import cv2
import os

def get_igloo_stage(ram_77_val):
    return 0 if ram_77_val == 255 else ram_77_val + 1

def main():
    data_path = "probing_data/Frostbite_20260822_201417.npz"
    print(f"데이터 로드 중: {data_path}")
    data = np.load(data_path)
    obs = data['obs']
    ram = data['ram']
    stages = np.array([get_igloo_stage(r[77]) for r in ram])

    out_dir = "frostbite_igloo_templates"
    os.makedirs(out_dir, exist_ok=True)

    print("템플릿 이미지 추출 중...")
    for stg in range(17):
        idxs = np.where(stages == stg)[0]
        if len(idxs) > 0:
            # 약간 안정화된 뒤의 프레임 사용 (가능하면 5프레임 뒤)
            idx = min(idxs[0] + 5, len(obs)-1)
            if stages[idx] != stg:
                idx = idxs[0]
                
            crop = obs[idx, 10:17, 42:58, :].copy()
            
            # 실제 가치 평가에 사용할 Crop 템플릿 (손실 없는 PNG 형식)
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{out_dir}/stage_{stg:02d}.png", crop_bgr)
            
            # 육안 검증을 위해 전체 화면(빨간 박스 포함)도 256x256 크기로 함께 저장
            full_img = obs[idx].copy()
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.rectangle(full_img, (42, 10), (58, 17), (0, 0, 255), 1)
            full_large = cv2.resize(full_img, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{out_dir}/context_stage_{stg:02d}.png", full_large)
            
            print(f"  Stage {stg:02d} 추출 완료 (프레임 인덱스: {idx})")
        else:
            print(f"  경고: Stage {stg:02d} 데이터를 찾지 못했습니다!")

    print(f"\n완료! '{out_dir}' 폴더에서 이미지를 확인하세요.")
    print("문제가 있는 이미지가 있다면, 그림판 등으로 직접 편집해서 덮어씌우면 됩니다.")

if __name__ == "__main__":
    main()
